"""Select and verify cited source sentences before rendering the demo answer."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx


class ExplanationRejected(ValueError):
    """The draft could not be tied to the retrieved passage text."""


@dataclass(frozen=True, slots=True)
class Prepared:
    result: dict
    question: str
    instruction: str
    version: str


@dataclass(frozen=True, slots=True)
class Draft:
    value: object
    generation: dict
    corrected: bool = False


ANSWER_STYLES = {"concise", "detailed", "steps"}


def _sentences(text):
    return re.split(r"(?<=[.!?])\s+", text)


def _claims(citations, evidence):
    if not isinstance(citations, list) or not 1 <= len(citations) <= 6:
        raise ExplanationRejected("Select one to six evidence sentences.")
    passages = {item["source_id"]: _sentences(item["text"]) for item in evidence}
    claims, seen = [], set()
    for citation in citations:
        if not isinstance(citation, dict) or set(citation) != {"source_id", "quote"}:
            raise ExplanationRejected("A citation requires a source ID and quote.")
        source_id, quote = citation["source_id"], citation["quote"]
        if not isinstance(source_id, str) or not isinstance(quote, str) or quote not in passages.get(source_id, ()):
            raise ExplanationRejected("Each quote must match a retrieved sentence verbatim.")
        if (source_id, quote) in seen:
            raise ExplanationRejected("Repeat citations only once.")
        seen.add((source_id, quote))
        claims.append({"text": quote, "citation_ids": [source_id], "supports": [{"id": source_id, "quote": quote}]})
    return claims


def _render(claims, style):
    lines = [f"{claim['text']} [{claim['citation_ids'][0]}]" for claim in claims]
    if style == "steps":
        return "\n".join(f"{number}. {line}" for number, line in enumerate(lines, start=1))
    return ("\n" if style == "concise" else "\n\n").join(lines)


def _answer_style(result):
    memory = result.get("conversation_context", {}).get("long_term_memory", ())
    for entry in memory:
        if entry.get("field") == "answer_style" and entry.get("value") in ANSWER_STYLES:
            return entry["value"]
    return "detailed"


def verify_result(original, candidate):
    """Keep the retrieval outcome and render every published line from verified evidence."""
    protected = ("status", "policy_unavailable", "sources", "evidence", "assistant_reply")
    if not isinstance(candidate, dict) or any(candidate.get(key) != original.get(key) for key in protected):
        raise ExplanationRejected("The draft changed the retrieval result.")
    claims = candidate.get("explanation_claims")
    if claims is None:
        if any(candidate.get(key) != original.get(key) for key in ("answer", "student_answer")):
            raise ExplanationRejected("The answer changed without a citation.")
        return

    try:
        citations = [
            {"source_id": claim["supports"][0]["id"], "quote": claim["supports"][0]["quote"]}
            for claim in claims
        ]
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise ExplanationRejected("Invalid citations.") from exc
    canonical = _claims(citations, original["evidence"])
    style = candidate.get("answer_style")
    if not isinstance(style, str) or style not in ANSWER_STYLES:
        raise ExplanationRejected("Unknown answer format.")
    answer = _render(canonical, style)
    if canonical != claims or any(candidate.get(key) != answer for key in ("answer", "student_answer")):
        raise ExplanationRejected("The answer must match the verified quotations.")


class NaturalExplainer:
    """Use deterministic citations offline, or let a configured model select the sentences."""

    prompt_version = "lab-v1"

    def __init__(self, *, api_key=None, base_url=None, model=None, transport=None):
        self.api_key = os.getenv("LLM_API_KEY", "") if api_key is None else api_key
        self.base_url = (os.getenv("LLM_BASE_URL", "") if base_url is None else base_url).rstrip("/")
        self.model = os.getenv("LLM_MODEL", "") if model is None else model
        self.transport = transport
        supplied = (self.api_key, self.base_url, self.model)
        if any(supplied) and not all(supplied):
            raise ValueError("Set LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL together.")
        self.remote_enabled = all(supplied)
        if self.remote_enabled:
            parts = urlsplit(self.base_url)
            loopback = parts.hostname in {"localhost", "127.0.0.1", "::1"}
            if ((parts.scheme != "https" and not (parts.scheme == "http" and loopback))
                    or not parts.hostname or parts.username or parts.query or parts.fragment):
                raise ValueError("Use an HTTPS OpenAI-compatible base URL, or HTTP for a local model.")

    def prepare(self, result, message, *, project_instruction="", project_version=None):
        return Prepared(result, message, project_instruction, project_version or self.prompt_version)

    def generate(self, prepared, *, correction=None):
        evidence = prepared.result["evidence"]
        generation = {
            "requested_mode": "explain", "mode": "original", "remote_called": False,
            "fallback": False, "validation_state": "skipped", "prompt_version": prepared.version,
        }
        if not evidence:
            return Draft(None, generation)
        if not self.remote_enabled:
            citations = [
                {"source_id": entry["source_id"], "quote": sentence}
                for entry in evidence for sentence in _sentences(entry["text"])
            ]
            return Draft({"citations": citations}, generation)

        instruction = (
            'Select complete evidence sentences that answer the question. Return only JSON in the form '
            '{"citations":[{"source_id":"<id>","quote":"<one sentence copied verbatim>"}]}.'
        )
        if prepared.instruction:
            instruction = f"{prepared.instruction}\n\n{instruction}"
        if correction:
            instruction += "\nThe previous citations failed validation. Copy the complete source sentence exactly."
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps({
                    "question": prepared.question,
                    "evidence": [{key: entry[key] for key in ("source_id", "text")} for entry in evidence],
                }, ensure_ascii=False)},
            ],
            "max_tokens": 400,
        }
        generation.update(remote_called=True, model=self.model)
        try:
            with httpx.Client(timeout=20, transport=self.transport, follow_redirects=False) as client:
                response = client.post(f"{self.base_url}/chat/completions", json=payload,
                                       headers={"Authorization": f"Bearer {self.api_key}"})
                response.raise_for_status()
                body = response.json()
            content = body["choices"][0]["message"]["content"]
            usage = body.get("usage")
            if isinstance(usage, dict):
                generation["usage"] = {
                    key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    if type(usage.get(key)) is int and usage[key] >= 0
                }
        except (httpx.HTTPError, IndexError, KeyError, TypeError, ValueError) as exc:
            generation.update(fallback=True, fallback_reason=type(exc).__name__)
            return Draft(None, generation, bool(correction))
        try:
            if isinstance(content, str):
                fenced = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", content, re.DOTALL)
                if fenced:
                    content = fenced.group(1)
            value = json.loads(content)
        except (TypeError, ValueError):
            value = None
        return Draft(value, generation, bool(correction))

    def verify(self, prepared, draft):
        generation = dict(draft.generation)
        result = {**prepared.result, "generation": generation}
        if not prepared.result["evidence"] or generation["fallback"]:
            return result
        try:
            citations = draft.value.get("citations") if isinstance(draft.value, dict) else None
            claims = _claims(citations, prepared.result["evidence"])
            style = _answer_style(prepared.result)
            answer = _render(claims, style)
            explained = {**result, "answer": answer, "student_answer": answer,
                         "explanation_claims": claims, "answer_style": style}
            verify_result(prepared.result, explained)
        except ExplanationRejected:
            generation.update(fallback=True, fallback_reason="ExplanationRejected", validation_state="rejected")
            if not draft.corrected:
                generation["correction"] = "grounding"
            return result
        generation.update(mode="explain", validation_state="passed")
        return explained

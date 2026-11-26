"""Runtime graph and role contracts for one policy-assistant turn."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from operator import add
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command
from langsmith import tracing_context

from .tracing import model_usage, trace_event


@dataclass(frozen=True, slots=True)
class PromptPolicy:
    version: str | None = None
    instruction: str = ""
    nodes: dict[str, str] = field(default_factory=dict)
    digest: str | None = None

    def for_role(self, role: str) -> str:
        parts = [self.instruction, self.nodes.get("generation", "")]
        if role == "revision":
            parts.append(self.nodes.get("revision", ""))
        return "\n".join(part for part in parts if part)


def load_prompt_policy(path: Path | None) -> PromptPolicy:
    if path is None:
        return PromptPolicy()
    try:
        with path.open("rb") as handle:
            raw = handle.read(16385)
        if len(raw) > 16384:
            raise ValueError("prompt policy too large")
        data = json.loads(raw.decode("utf-8-sig"))
        version = data.get("version")
        instruction = data.get("instruction", "")
        nodes = data.get("nodes", {})
        if (not isinstance(version, str) or not re.fullmatch(r"[a-zA-Z0-9._-]{1,64}", version)
                or not isinstance(instruction, str) or not isinstance(nodes, dict)
                or nodes.keys() - {"generation", "revision"}
                or any(not isinstance(value, str) for value in nodes.values())):
            raise ValueError("invalid prompt policy")
        digest = hashlib.sha256(raw).hexdigest()
        return PromptPolicy(version, instruction.strip(), {key: text.strip() for key, text in nodes.items()}, digest)
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        trace_event("workflow_prompt", status="invalid", error_type=type(exc).__name__)
        return PromptPolicy()


class WorkflowState(TypedDict, total=False):
    original: dict
    conversation: dict
    prepared: Any
    draft: Any
    verified: dict
    published: dict
    attempt: int
    attempts: Annotated[list[dict], add]
    decision: str


@dataclass(frozen=True, slots=True)
class WorkflowContext:
    retrieve: Callable[[], dict]
    prepare_dialogue: Callable[[dict], dict]
    explainer: Any
    message: str
    mode: str
    eligible: bool
    revision_retry: bool
    prompt: PromptPolicy
    recall_memory: Callable[[dict], list[dict]] | None
    persist_turn: Callable[[dict], None] | None

    @property
    def split_explainer(self) -> bool:
        return all(callable(getattr(self.explainer, name, None)) for name in ("prepare", "generate", "verify"))


def _retrieve(state: WorkflowState, runtime: Runtime[WorkflowContext]) -> dict:
    return {"original": runtime.context.retrieve()}


def _prepare_dialogue(
    state: WorkflowState, runtime: Runtime[WorkflowContext],
) -> Command[Literal["recall_memory", "generation_agent", "publish"]]:
    context = runtime.context
    original = state["original"]
    conversation = context.prepare_dialogue(original)
    if not context.eligible or original.get("status") == "error":
        trace_event("model", status="skipped", reason="original_mode" if context.mode == "original" else "not_applicable")
        return Command(update={"conversation": conversation, "decision": "original"}, goto="publish")
    can_recall = context.recall_memory is not None and original.get("status") != "no_evidence"
    next_node = "recall_memory" if can_recall else "generation_agent"
    return Command(update={"conversation": conversation, "decision": "generate"}, goto=next_node)


def _recall_memory(state: WorkflowState, runtime: Runtime[WorkflowContext]) -> dict:
    recall = runtime.context.recall_memory
    assert recall is not None
    remembered = recall(state["original"])
    if remembered:
        return {"conversation": {**state["conversation"], "long_term_memory": remembered}}
    return {}


def _generate(state: WorkflowState, runtime: Runtime[WorkflowContext]) -> dict:
    context = runtime.context
    attempt = state.get("attempt", 0) + 1
    role = "revision" if attempt > 1 else "generation"
    source = {**copy.deepcopy(state["original"]), "conversation_context": state["conversation"]}
    try:
        if context.split_explainer:
            prepared = context.explainer.prepare(source, context.message,
                project_instruction=context.prompt.for_role(role), project_version=context.prompt.version)
            draft = context.explainer.generate(prepared, correction="grounding" if attempt > 1 else None)
            return {"prepared": prepared, "draft": draft, "attempt": attempt}
        return {"draft": context.explainer.explain(source, context.message), "attempt": attempt}
    except Exception as exc:
        trace_event("model", status="error", error_type=type(exc).__name__)
        return {"prepared": None, "draft": exc, "attempt": attempt}


def _fallback(original, reason: str, telemetry: dict) -> dict:
    generation = {"requested_mode": "explain", "mode": "original", "remote_called": bool(telemetry.get("remote_called")),
                  "fallback": True, "fallback_reason": reason, "validation_state": "rejected"}
    usage = model_usage(telemetry.get("usage"))
    if usage:
        generation["usage"] = usage
    return {**original, "generation": generation}


def _verify(
    state: WorkflowState, runtime: Runtime[WorkflowContext],
) -> Command[Literal["generation_agent", "publish"]]:
    from .explanation import ExplanationRejected, verify_result

    context = runtime.context
    original = state["original"]
    draft = state.get("draft")
    telemetry = getattr(draft, "generation", {}) or {}
    try:
        if isinstance(draft, Exception):
            candidate = _fallback(original, type(draft).__name__, telemetry)
        elif context.split_explainer:
            candidate = context.explainer.verify(state["prepared"], draft)
        else:
            candidate = {**original, **draft} if draft else original
        telemetry = {**candidate.get("generation", {}), **telemetry}
        verify_result(original, candidate)
    except ExplanationRejected:
        candidate = _fallback(original, "workflow_verification_rejected", telemetry)
    except Exception as exc:
        candidate = _fallback(original, type(exc).__name__, telemetry)
    generation = candidate.get("generation", {})
    can_revise = (context.revision_retry and context.split_explainer and state.get("attempt") == 1
                  and generation.get("fallback") and generation.get("validation_state") == "rejected"
                  and generation.get("correction") == "grounding")
    decision = "revise" if can_revise else "fallback" if generation.get("fallback") else "publish"
    attempt = {"attempt": state["attempt"], "remote_called": bool(telemetry.get("remote_called") or generation.get("remote_called")),
               "usage": model_usage(telemetry.get("usage") or generation.get("usage")), "outcome": decision}
    trace_event("workflow_attempt", status="completed", **attempt)
    return Command(update={"verified": candidate, "decision": decision, "attempts": [attempt]},
                   goto="generation_agent" if can_revise else "publish")


def _finish(state: WorkflowState, runtime: Runtime[WorkflowContext]) -> Command[Literal["persist_turn", "__end__"]]:
    source = dict(state.get("verified", state["original"]))
    source.pop("conversation_context", None)
    if attempts := state.get("attempts"):
        calls = sum(attempt["remote_called"] for attempt in attempts)
        source["generation"] = {**source["generation"], "remote_called": bool(calls),
                                "workflow_attempts": len(attempts), "workflow_model_calls": calls}
    if source["generation"].get("mode") == "explain" and not source.get("policy_unavailable"):
        reply = source.get("assistant_reply")
        if source.get("explanation_claims"):
            source["student_answer"] = source["answer"]
        elif reply:
            if source["status"] != "needs_info":
                source["student_answer"] = reply + "\n\n" + source["student_answer"]
            elif not source.get("award_choices"):
                source["student_answer"] = reply
    next_node = "persist_turn" if runtime.context.persist_turn is not None else END
    return Command(update={"published": source}, goto=next_node)


def _persist_turn(state: WorkflowState, runtime: Runtime[WorkflowContext]) -> dict:
    persist = runtime.context.persist_turn
    assert persist is not None
    persist(state["published"])
    return {}


class AgentWorkflow:
    def __init__(self, *, prompt_path: Path | None = None, revision_retry: bool = False):
        self.prompt_path = prompt_path
        self.revision_retry = revision_retry
        graph = StateGraph(WorkflowState, context_schema=WorkflowContext)
        graph.add_node("retrieval_agent", _retrieve)
        graph.add_node("prepare_dialogue", _prepare_dialogue)
        graph.add_node("recall_memory", _recall_memory)
        graph.add_node("generation_agent", _generate)
        graph.add_node("verification_agent", _verify)
        graph.add_node("publish", _finish)
        graph.add_node("persist_turn", _persist_turn)
        graph.add_edge(START, "retrieval_agent")
        graph.add_edge("retrieval_agent", "prepare_dialogue")
        graph.add_edge("recall_memory", "generation_agent")
        graph.add_edge("generation_agent", "verification_agent")
        graph.add_edge("persist_turn", END)
        self.graph = graph.compile()

    def run(self, retrieve, prepare_dialogue, *, explainer, message, mode, eligible,
            recall_memory: Callable[[dict], list[dict]] | None = None,
            persist_turn: Callable[[dict], None] | None = None) -> dict:
        prompt = load_prompt_policy(self.prompt_path) if eligible else PromptPolicy()
        context = WorkflowContext(retrieve, prepare_dialogue, explainer, message, mode, eligible,
                                  self.revision_retry, prompt, recall_memory, persist_turn)
        version = prompt.version or (getattr(explainer, "prompt_version", None) if eligible else None)
        starts = {}
        published = None
        with tracing_context(enabled=False):
            for task in self.graph.stream({"attempt": 0}, config={"recursion_limit": 10}, context=context,
                                          stream_mode="tasks"):
                # Keep only stage metadata. Native task payloads can contain student input and raw drafts.
                node = task["name"]
                if "input" in task:
                    started = time.perf_counter()
                    attempt = task["input"].get("attempt", 0) + (node == "generation_agent")
                    metadata = {"node": node, "attempt": attempt, "prompt_version": version, "prompt_sha256": prompt.digest}
                    starts[task["id"]] = (started, metadata)
                    trace_event("workflow_node", started=started, status="started", **metadata)
                    continue
                started, metadata = starts.pop(task["id"])
                if (error := task["error"]) is not None:
                    trace_event("workflow_node", started=started, status="error", error_type=type(error).__name__, **metadata)
                    continue
                updates = task["result"]
                outcome = updates.get("decision") or (updates.get("original") or {}).get("status")
                trace_event("workflow_node", started=started, status="completed", outcome=outcome, **metadata)
                if node == "publish":
                    published = updates["published"]
        if published is None:
            raise RuntimeError("Policy workflow did not publish a response")
        return published

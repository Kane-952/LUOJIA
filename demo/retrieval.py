"""Hybrid retrieval over a fictional equipment lab's policy passages."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .graph_cache import BoundedGraphCache
from .memory import embed


@dataclass(frozen=True, slots=True)
class Passage:
    source_id: str
    title: str
    text: str
    terms: frozenset[str]


PASSAGES = (
    Passage("LAB-LOAN", "Equipment loan periods",
            "A standard equipment loan lasts 72 hours. You may renew it once for another 48 hours before it is due.",
            frozenset("borrow borrowing loan loans renewal renew".split())),
    Passage("LAB-DEPOSIT", "Field recorder deposit",
            "A field recorder requires a refundable deposit of $25.",
            frozenset("deposit deposits refundable refund recorder recorders".split())),
    Passage("LAB-RETURN", "Equipment return locations",
            "Return equipment to the service desk during opening hours. After hours, use the return lockers outside room M-12.",
            frozenset("return returns returning locker lockers closing".split())),
    Passage("LAB-TRAINING", "Machine orientation",
            "Complete the 30-minute orientation before using the laser cutter.",
            frozenset("orientation training laser cutter certification".split())),
    Passage("LAB-REPAIR", "Equipment fault reports",
            "Report damaged equipment to the service desk before the next shift.",
            frozenset("damage damaged broken fault maintenance malfunction repair".split())),
)
NO_EVIDENCE = "The lab policy passages do not answer this question."
_CACHE = BoundedGraphCache("fictional_lab_retrieval", 64, ttl=120)
_VECTORS = embed([f"{passage.title} {passage.text}" for passage in PASSAGES])


def retrieve(question: str) -> dict:
    """Require a policy term, then rank the matching passages by lexical and vector scores."""
    query = " ".join(question.casefold().split())
    key = hashlib.sha256(query.encode("utf-8")).hexdigest()
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    terms = set(re.findall(r"[a-z0-9]+", query))
    overlaps = [len(terms & passage.terms) for passage in PASSAGES]
    ranked = []
    if any(overlaps):
        vector = embed([query])[0]
        for passage, lexical, stored in zip(PASSAGES, overlaps, _VECTORS, strict=True):
            if lexical:
                similarity = sum(a * b for a, b in zip(vector, stored, strict=True))
                score = 0.7 * min(1.0, lexical / 3) + 0.3 * similarity
                ranked.append((score, passage))
    ranked.sort(key=lambda pair: (-pair[0], pair[1].source_id))
    evidence = [
        {"source_id": passage.source_id, "text": passage.text, "score": round(score, 4)}
        for score, passage in ranked[:3]
    ]
    sources = [
        {"source_id": passage.source_id, "title": passage.title}
        for _, passage in ranked[:3]
    ]
    answer = "\n\n".join(f"{entry['text']} [{entry['source_id']}]" for entry in evidence) or NO_EVIDENCE
    result = {
        "status": "answered" if evidence else "no_evidence",
        "answer": answer,
        "student_answer": answer,
        "policy_unavailable": not evidence,
        "evidence": evidence,
        "sources": sources,
        "generation": {
            "requested_mode": "original", "mode": "original", "remote_called": False, "fallback": False,
        },
    }
    _CACHE.set(key, result)
    return result

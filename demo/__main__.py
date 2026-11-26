"""Command-line entry points for the public LUOJIA implementation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .agent_workflow import AgentWorkflow
from .explanation import NaturalExplainer
from .memory import open_memory
from .retrieval import retrieve
from .tracing import TraceStore, TurnTrace


FOLLOW_UPS = {"what about that", "tell me more", "can you elaborate"}


def ask(args, parser):
    question = args.question.strip()
    if not question or len(question) > 1_000:
        parser.error("Enter a question with 1 to 1,000 characters.")
    owner_and_session = json.dumps([args.owner, args.session], ensure_ascii=False)
    thread = hashlib.sha256(owner_and_session.encode()).hexdigest()
    workflow = AgentWorkflow(prompt_path=args.prompt, revision_retry=True)
    explainer = NaturalExplainer() if args.mode == "explain" else None
    trace = TurnTrace(TraceStore(args.state_dir), args.mode, "cli")
    with open_memory(args.state_dir / "memory.sqlite3") as memory, trace.activate():
        history = memory.history(thread)
        follow_up = question.casefold().rstrip("?.!") in FOLLOW_UPS
        query = history[-1]["question"] if follow_up and history else question
        try:
            result = workflow.run(
                lambda: retrieve(query),
                lambda _: {},
                explainer=explainer,
                message=question,
                mode=args.mode,
                eligible=args.mode == "explain",
                recall_memory=lambda _: memory.recall(args.owner, question),
                persist_turn=lambda reply: memory.save_turn(thread, query, reply["answer"]),
            )
        except Exception as exc:
            trace.finish(status="error", error_type=type(exc).__name__)
            raise
        trace.finish(status=result["status"], source_ids=[source["source_id"] for source in result["sources"]])
    if args.json:
        print(json.dumps({**result, "trace_id": trace.payload["trace_id"]}, ensure_ascii=False, indent=2))
    else:
        print(result["answer"])


def evaluate():
    workflow = AgentWorkflow(revision_retry=True)
    explainer = NaturalExplainer(api_key="", base_url="", model="")
    cases = (
        ("How long can I borrow equipment?", "LAB-LOAN", "72 hours"),
        ("What deposit is required for a field recorder?", "LAB-DEPOSIT", "$25"),
        ("Where do I return equipment after hours?", "LAB-RETURN", "M-12"),
        ("What is the capital of Mars?", None, None),
    )
    passed = 0
    for question, source, term in cases:
        result = workflow.run(lambda: retrieve(question), lambda _: {}, explainer=explainer,
                              message=question, mode="explain", eligible=True)
        valid = (
            result["status"] == "answered" and term in result["answer"]
            and any(source in claim["citation_ids"] for claim in result.get("explanation_claims", []))
            and result["generation"].get("validation_state") == "passed"
        ) if source else result["status"] == "no_evidence" and not result["sources"]
        print(f"{'PASS' if valid else 'FAIL'}  {question}")
        passed += int(valid)
    print(f"{passed}/{len(cases)} passed")
    return 0 if passed == len(cases) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=Path(".luojia"))
    commands = parser.add_subparsers(dest="command", required=True)
    question = commands.add_parser("ask", help="answer a fictional lab policy question")
    question.add_argument("question")
    question.add_argument("--owner", default="local")
    question.add_argument("--session", default="default")
    question.add_argument("--mode", choices=("original", "explain"), default="explain")
    question.add_argument("--prompt", type=Path, help="optional JSON instruction file")
    question.add_argument("--json", action="store_true")
    remember = commands.add_parser("remember", help="save a local answer-format preference")
    remember.add_argument("preference", choices=("concise", "detailed", "steps"))
    remember.add_argument("--owner", default="local")
    commands.add_parser("evaluate", help="run the four offline reference cases")
    trace = commands.add_parser("trace", help="view a saved node trace")
    trace.add_argument("trace_id")
    args = parser.parse_args()
    try:
        if args.command == "ask":
            ask(args, parser)
        elif args.command == "remember":
            with open_memory(args.state_dir / "memory.sqlite3") as memory:
                memory.remember(args.owner, args.preference)
            print(f"Saved answer style: {args.preference}.")
        elif args.command == "trace":
            row = TraceStore(args.state_dir).read(args.trace_id)
            if row is None:
                parser.error("Trace not found.")
            print(json.dumps(row, ensure_ascii=False, indent=2))
        else:
            return evaluate()
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

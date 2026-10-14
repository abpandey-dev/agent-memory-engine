"""`memory` command line.

  memory show-data --persona 0                 print a synthetic persona's conversations and questions
  memory demo --persona 0 --model claude-haiku-5-5   build memory for one persona and answer its questions
  memory bench --model claude-haiku-5-5 --personas 6 [--filler 24] [--methods memory full_context]
  memory ask "Where do I live?" --db me.db --user me   answer from a saved memory database
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from agent_memory.bench import METHODS, Runner, is_correct, results_json, summarize
from agent_memory.dataset import dataset, make_persona
from agent_memory.engine import MemoryEngine, format_hits
from agent_memory.llm import DEFAULT_MODEL, LLM
from agent_memory.store import MemoryStore


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="memory", description="Long-term memory engine for AI agents")
    sub = p.add_subparsers(dest="cmd", required=True)

    show = sub.add_parser("show-data", help="Print a synthetic persona (no API calls)")
    show.add_argument("--persona", type=int, default=0)
    show.add_argument("--filler", type=int, default=0, help="extra small-talk-only sessions")

    demo = sub.add_parser("demo", help="Build memory for one persona and answer its questions")
    demo.add_argument("--persona", type=int, default=0)
    demo.add_argument("--model", default=DEFAULT_MODEL)

    bench = sub.add_parser("bench", help="Compare memory against no memory, full context and chunk RAG")
    bench.add_argument("--model", default=DEFAULT_MODEL)
    bench.add_argument("--personas", type=int, default=6)
    bench.add_argument("--filler", type=int, default=0, help="extra small-talk-only sessions per persona")
    bench.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    bench.add_argument("--out", default="reports")

    ask = sub.add_parser("ask", help="Answer a question from a saved memory database")
    ask.add_argument("question")
    ask.add_argument("--db", default="memory.db")
    ask.add_argument("--user", default="default")
    ask.add_argument("--model", default=DEFAULT_MODEL)

    args = p.parse_args(argv)

    if args.cmd == "show-data":
        persona = make_persona(args.persona, args.filler)
        for s in persona.sessions:
            print(f"\n=== {s.id} ({s.date})")
            for m in s.messages:
                print(f"  {m['role']:<9} {m['content']}")
        print(f"\nQuestions (asked on {persona.today}):")
        for q in persona.questions:
            exp = " + ".join("/".join(g) for g in q.expected) if q.known else "I don't know"
            print(f"  [{q.type}] {q.question}  -> {exp}")
        return 0

    if args.cmd == "demo":
        persona = make_persona(args.persona)
        llm = LLM(args.model)
        engine = MemoryEngine(llm=llm)
        for s in persona.sessions:
            r = engine.ingest(persona.user_id, s.messages, s.date, s.id)
            print(f"{s.id}: +{len(r.added)} added, {len(r.updated)} updated {r.updated}, {len(r.skipped)} skipped")
        print("\nMemory:")
        for m in engine.store.all(persona.user_id):
            flag = "" if m.current else f"   [superseded {m.valid_until} by {m.superseded_by}]"
            print(f"  {m.id} {m.text}{flag}")
        print()
        correct = 0
        for q in persona.questions:
            a = engine.answer(persona.user_id, q.question, persona.today)
            ok = is_correct(q, a.known, a.answer)
            correct += ok
            print(f"{'✓' if ok else '✗'} {q.question}\n    {a.answer}  (memories: {', '.join(a.memory_ids) or '-'})")
        cost = llm.usage.cost(args.model)
        print(f"\n{correct}/{len(persona.questions)} correct; {llm.usage.calls} calls, "
              f"{llm.usage.input_tokens:,}+{llm.usage.output_tokens:,} tokens"
              + (f", ${cost:.4f}" if cost is not None else ""))
        return 0

    if args.cmd == "bench":
        runner = Runner(args.model)
        personas = dataset(args.personas, args.filler)
        results = runner.run(personas, args.methods)
        table = summarize(results, args.methods, args.model, runner.ingest_usage)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        name = f"bench_{args.model}_p{args.personas}_f{args.filler}"
        (out / f"{name}.json").write_text(json.dumps(results_json(results), indent=1), encoding="utf-8")
        (out / f"{name}.md").write_text(table, encoding="utf-8")
        print(table)
        for r in results:
            if not r.correct:
                print(f"  miss [{r.method}] {r.question_id} ({r.type}): {r.answer or r.error}")
        return 0

    if args.cmd == "ask":
        engine = MemoryEngine(store=MemoryStore(args.db), model=args.model)
        hits = engine.recall(args.user, args.question)
        print("Recalled:\n" + format_hits(hits))
        a = engine.answer(args.user, args.question, date.today().isoformat())
        print(f"\n{a.answer}  (memories: {', '.join(a.memory_ids) or '-'})")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

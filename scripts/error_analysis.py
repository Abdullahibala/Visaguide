"""Rank the worst failures of an answer-evaluation run and tag a likely failure type.

    python scripts/error_analysis.py --run eval/results/answers_test_... --n 10

Severity (most harmful first) - a wrong answer stated with confidence is worse
than a refusal, because the student may act on it:
  5  unanswerable question answered with coverage "full"  (possible hallucination)
  4  answerable, verdict incorrect, coverage "full"       (confidently wrong)
  3  any unsupported claim (faithfulness < 1)
  2  answerable, refused (coverage "none")                (false refusal)
  1  answerable, verdict partial / incorrect but hedged   (incomplete)

Automatic type tags, which a human then checks:
  retrieval_miss     no retrieved chunk overlaps a gold passage
  evidence_ignored   gold was retrieved, but the answer is still wrong or incomplete
  unsupported_claim  the judge found claims the sources don't state
  answered_unanswerable
The output is a starting point: the final grouping in docs/ is written after
reading every case (some "retrieval misses" are really incomplete gold labels).
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from visaguide.config import ROOT
from visaguide.evaluation.retrieval_metrics import matches


def severity(r: dict) -> int:
    if not r["answerable"] and r["coverage"] == "full":
        return 5
    if r["answerable"] and r["verdict"] == "incorrect" and r["coverage"] == "full":
        return 4
    if r["faithfulness"] is not None and r["faithfulness"] < 1:
        return 3
    if r["answerable"] and r["coverage"] == "none":
        return 2
    if r["answerable"] and r["verdict"] in ("partial", "incorrect"):
        return 1
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    def load(name):
        with (args.run / name).open(encoding="utf-8") as f:
            return {r["id"]: r for r in map(json.loads, f)}

    rows = load("per_question.jsonl")
    answers, judged = load("answers.jsonl"), load("judgments.jsonl")
    with (ROOT / "eval" / "testset.jsonl").open(encoding="utf-8") as f:
        qs = {q["id"]: q for q in map(json.loads, f)}

    cases = []
    for qid, r in rows.items():
        sev = severity(r)
        if sev == 0:
            continue
        a, q, j = answers[qid], qs[qid], judged.get(qid, {})
        spans = [SimpleNamespace(**{k: x[k] for k in ("doc_id", "start", "end")})
                 for x in a["retrieved"]]
        gold_retrieved = any(matches(s, g) for s in spans for g in q["gold"])
        unsupported = [c["claim"] for c in ((j.get("faithfulness") or {}).get("claims") or [])
                       if not c["supported_by_any"]]
        tags = []
        if not q["answerable"] and r["coverage"] == "full":
            tags.append("answered_unanswerable")
        if q["answerable"] and not gold_retrieved:
            tags.append("retrieval_miss")
        if q["answerable"] and gold_retrieved and r["verdict"] != "correct":
            tags.append("evidence_ignored")
        if unsupported:
            tags.append("unsupported_claim")
        cases.append({"id": qid, "severity": sev, "tags": tags, "type": q["type"],
                      "coverage": r["coverage"], "verdict": r["verdict"],
                      "question": q["question"], "reference": q["reference_answer"],
                      "answer": a["answer"], "unsupported_claims": unsupported,
                      "judge": (j.get("correctness") or {}).get("explanation", "")})

    cases.sort(key=lambda c: (-c["severity"], c["id"]))
    worst = cases[:args.n]
    with (args.run / "worst_failures.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(worst, f, indent=2, ensure_ascii=False)
    print(f"{len(cases)} failing questions; worst {len(worst)}:\n")
    for c in worst:
        print(f"[{c['severity']}] {c['id']} {c['type']:11s} coverage={c['coverage']:7s} "
              f"verdict={c['verdict']} tags={c['tags']}")
        print(f"    Q:   {c['question']}")
        print(f"    REF: {c['reference'][:200]}")
        print(f"    ANS: {' '.join(c['answer'].split())[:400]}")
        for u in c["unsupported_claims"]:
            print(f"    UNSUPPORTED: {u[:160]}")
        if c["judge"]:
            print(f"    JUDGE: {c['judge'][:220]}")
        print()


if __name__ == "__main__":
    main()

"""Create a BLIND human-labelling sheet from an answer-evaluation run.

    python scripts/make_label_sheet.py --run eval/results/answers_test_... --n 20

Writes eval/human_labels.yaml. It deliberately does NOT show the judge's verdicts:
seeing them would anchor you, and the agreement number would be inflated.
Fill in `faithful` and `correctness` for every item, then run judge_agreement.py.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

from visaguide.config import ROOT, load_config

OUT = ROOT / "eval" / "human_labels.yaml"

INSTRUCTIONS = """\
# Human labels for checking the LLM judge. Label BEFORE looking at the judge's output.
#
# faithful:    yes = every factual claim in `answer` is stated in `sources`
#              no  = at least one claim is not stated there (even if true in reality)
# correctness: compare `answer` with `reference_answer`
#              correct   = all key facts, nothing contradicting
#              partial   = some key facts / vaguer, nothing contradicting
#              incorrect = contradicts a key fact, wrong number/rule, or doesn't answer
#              (leave as n/a for unanswerable questions)
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if OUT.exists():
        sys.exit(f"{OUT.relative_to(ROOT)} exists; delete it first (don't overwrite labels).")

    cfg = load_config()
    run_cfg = json.loads((args.run / "run_config.json").read_text(encoding="utf-8"))
    with (cfg.paths.index_dir / run_cfg["chunking"] / "chunks.jsonl").open(encoding="utf-8") as f:
        chunk_text = {c["chunk_id"]: c["text"] for c in map(json.loads, f)}
    with (ROOT / "eval" / "testset.jsonl").open(encoding="utf-8") as f:
        qs = {q["id"]: q for q in map(json.loads, f)}
    with (args.run / "answers.jsonl").open(encoding="utf-8") as f:
        answers = [a for a in map(json.loads, f) if not a["refused"]]  # judge skips refusals

    sample = random.Random(args.seed).sample(answers, min(args.n, len(answers)))
    items = []
    for a in sorted(sample, key=lambda a: a["id"]):
        q = qs[a["id"]]
        items.append({
            "id": a["id"],
            "question": q["question"],
            "reference_answer": q["reference_answer"] if q["answerable"] else "(unanswerable)",
            "answer": a["answer"],
            "sources": {f"S{i}": chunk_text[r["chunk_id"]]
                        for i, r in enumerate(a["retrieved"], 1)},
            "faithful": "",
            "correctness": "" if q["answerable"] else "n/a",
        })
    body = yaml.safe_dump(items, sort_keys=False, allow_unicode=True, width=100)
    OUT.write_text(INSTRUCTIONS + "\n" + body, encoding="utf-8", newline="\n")
    print(f"Wrote {len(items)} items to {OUT.relative_to(ROOT)} (run: {args.run.name})")


if __name__ == "__main__":
    main()

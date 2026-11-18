"""Create a BLIND human-labelling sheet from an answer-evaluation run.

    python scripts/make_label_sheet.py --run eval/results/answers_test_... --n 20

Writes eval/human_labels.yaml. It deliberately does NOT show the judge's verdicts:
seeing them would anchor you, and the agreement number would be inflated.
Fill in `faithful` and `correctness` for every item, then run judge_agreement.py.
"""

import argparse
import csv
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
#              Shared definition (same as the judge rubric v3):
#              - NOT claims: the standard disclaimers ("not legal advice", "rules can
#                change", "confirm with the authority") and "the sources don't say X".
#              - Claims: descriptions of the sources ("this guide is from 2024") and
#                generalisations ("other cities may differ").
#              - Evidence: a source's title, url and retrieved date count, not only its text.
# note:        for every "no", name the unsupported sentence.
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
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--exclude", type=Path, nargs="*", default=[],
                    help="existing label files whose ids must not be sampled again")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out_path = args.out
    if out_path.exists():
        sys.exit(f"{out_path} exists; delete it first (don't overwrite labels).")

    cfg = load_config()
    run_cfg = json.loads((args.run / "run_config.json").read_text(encoding="utf-8"))
    with (cfg.paths.index_dir / run_cfg["chunking"] / "chunks.jsonl").open(encoding="utf-8") as f:
        chunk_text = {c["chunk_id"]: c["text"] for c in map(json.loads, f)}
    with (ROOT / "eval" / "testset.jsonl").open(encoding="utf-8") as f:
        qs = {q["id"]: q for q in map(json.loads, f)}
    with cfg.paths.manifest.open(encoding="utf-8") as f:
        retrieved_at = {r["doc_id"]: r["retrieved_at"] for r in csv.DictReader(f)}
    exclude = set()
    for path in args.exclude:
        exclude |= {x["id"] for x in yaml.safe_load(path.read_text(encoding="utf-8"))}
    with (args.run / "answers.jsonl").open(encoding="utf-8") as f:
        answers = [a for a in map(json.loads, f)
                   if not a["refused"] and a["id"] not in exclude]  # judge skips refusals

    sample = random.Random(args.seed).sample(answers, min(args.n, len(answers)))
    items = []
    for a in sorted(sample, key=lambda a: a["id"]):
        q = qs[a["id"]]
        items.append({
            "id": a["id"],
            "question": q["question"],
            "reference_answer": q["reference_answer"] if q["answerable"] else "(unanswerable)",
            "answer": a["answer"],
            # Same information the model and the judge saw: attributes AND text.
            # (The first sheet showed text only, so claims supported by a source's
            # title or date looked unsupported to the labeller.)
            "sources": {f"S{i}": {"title": r["title"], "url": r["url"],
                                  "retrieved": retrieved_at.get(r["doc_id"], "")[:10],
                                  "text": chunk_text[r["chunk_id"]]}
                        for i, r in enumerate(a["retrieved"], 1)},
            "faithful": "",
            "correctness": "" if q["answerable"] else "n/a",
            "note": "",
        })
    body = yaml.safe_dump(items, sort_keys=False, allow_unicode=True, width=100)
    out_path.write_text(INSTRUCTIONS + "\n" + body, encoding="utf-8", newline="\n")
    print(f"Wrote {len(items)} items to {out_path} (run: {args.run.name})")


if __name__ == "__main__":
    main()

"""How often does the LLM judge agree with your labels? (raw agreement + Cohen's kappa)

    python scripts/judge_agreement.py --run eval/results/answers_test_...
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from visaguide.config import ROOT
from visaguide.evaluation.agreement import cohen_kappa, confusion, raw_agreement

LABELS = ROOT / "eval" / "human_labels.yaml"


def report(name: str, human: list, judge: list) -> list[str]:
    lines = [f"### {name} (n = {len(human)})", ""]
    if not human:
        return lines + ["No labelled items.", ""]
    lines += [f"- Raw agreement: **{raw_agreement(human, judge):.2f}**",
              f"- Cohen's kappa: **{cohen_kappa(human, judge):.2f}**", "",
              "| Human | Judge | Count |", "|---|---|---|"]
    for (h, j), n in sorted(confusion(human, judge).items()):
        lines.append(f"| {h} | {j} | {n} |")
    return lines + [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--faithful-field", default="faithful",
                    help="label field to compare, e.g. faithful_shared_def")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    labels = yaml.safe_load(args.labels.read_text(encoding="utf-8"))
    ff = args.faithful_field
    missing = [x["id"] for x in labels if not x.get(ff) or not x["correctness"]]
    if missing:
        sys.exit(f"Unlabelled items: {missing}")
    with (args.run / "judgments.jsonl").open(encoding="utf-8") as f:
        judged = {j["id"]: j for j in map(json.loads, f)}

    fh, fj, ch, cj = [], [], [], []
    for x in labels:
        j = judged[x["id"]]
        claims = (j.get("faithfulness") or {}).get("claims") or []
        fh.append(str(x[ff]).lower())
        fj.append("yes" if all(c["supported_by_any"] for c in claims) else "no")
        if x["correctness"] != "n/a":
            ch.append(x["correctness"])
            cj.append((j.get("correctness") or {}).get("verdict", "missing"))

    out = ["## LLM judge vs human labels", "",
           f"Run: `{args.run.name}`. Labels: `{args.labels.name}`, field `{ff}`. "
           "Faithful (judge) = every claim supported_by_any.", ""]
    out += report("Faithfulness (yes/no)", fh, fj)
    caught = sum(h == "no" and j == "no" for h, j in zip(fh, fj, strict=True))
    out += [f"Judge recall on human-unfaithful answers: **{caught}/{fh.count('no')}**", ""]
    out += report("Correctness (correct/partial/incorrect)", ch, cj)
    text = "\n".join(out)
    suffix = "" if ff == "faithful" else f"_{ff}"
    (args.run / f"judge_agreement{suffix}.md").write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()

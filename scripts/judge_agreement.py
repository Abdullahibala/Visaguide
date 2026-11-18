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
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    labels = yaml.safe_load(LABELS.read_text(encoding="utf-8"))
    missing = [x["id"] for x in labels if not x["faithful"] or not x["correctness"]]
    if missing:
        sys.exit(f"Unlabelled items: {missing}")
    with (args.run / "judgments.jsonl").open(encoding="utf-8") as f:
        judged = {j["id"]: j for j in map(json.loads, f)}

    fh, fj, ch, cj = [], [], [], []
    for x in labels:
        j = judged[x["id"]]
        claims = (j.get("faithfulness") or {}).get("claims") or []
        fh.append(str(x["faithful"]).lower())
        fj.append("yes" if all(c["supported_by_any"] for c in claims) else "no")
        if x["correctness"] != "n/a":
            ch.append(x["correctness"])
            cj.append((j.get("correctness") or {}).get("verdict", "missing"))

    out = ["## LLM judge vs human labels", "",
           f"Run: `{args.run.name}`. Faithful (judge) = every claim supported_by_any.", ""]
    out += report("Faithfulness (yes/no)", fh, fj)
    out += report("Correctness (correct/partial/incorrect)", ch, cj)
    text = "\n".join(out)
    (args.run / "judge_agreement.md").write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()

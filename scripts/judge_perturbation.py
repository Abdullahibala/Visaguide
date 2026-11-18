"""Perturbation test: does the faithfulness judge catch unsupported claims we insert?

Why: with only a few unfaithful answers in a random sample, agreement statistics
can't tell whether the judge catches errors (prevalence paradox: 18 of 20 labelled
answers were faithful). So we create errors on purpose and measure recall directly.

For each sampled answer we make two corrupted copies:
  number   - one number in a cited sentence is changed (e.g. 140 -> 210).
             The most harmful error for a visa assistant: a wrong fee or deadline.
  addition - one plausible but unsupported sentence with a citation marker is added
             (a rule or deadline that no source states).
Detection = the judge marks the corrupted claim as unsupported. A flag on some other
claim doesn't count.

    python scripts/judge_perturbation.py --run eval/results/answers_test_... --n 20
"""

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

from visaguide.chunking import Chunk
from visaguide.config import load_config
from visaguide.evaluation.judge import judge_faithfulness
from visaguide.generation.llm import make_llm
from visaguide.generation.prompts import build_sources_block

ADDITIONS = [
    "This rule applies in the same way to students of every nationality [S1].",
    "The responsible office usually decides within two weeks of receiving the application [S1].",
    "This requirement was introduced in 2019 and has not changed since [S1].",
    "Students who miss this deadline can apply again after paying a late fee of 50 euros [S1].",
    "Part-time students are exempt from this requirement [S1].",
]
NUMBER = re.compile(r"(?<![\w\[S])(\d{1,3}(?:[.,]\d{3})*|\d+)(?![\w\]])")


def corrupt_number(answer: str, rng: random.Random) -> tuple[str, str] | None:
    """Change one number outside citation markers; return (new_answer, new_number)."""
    candidates = [m for m in NUMBER.finditer(answer)
                  if not re.match(r"S\d", answer[max(0, m.start() - 1):m.end()])]
    if not candidates:
        return None
    m = rng.choice(candidates)
    digits = int(re.sub(r"[.,]", "", m.group()))
    new = digits + max(1, round(digits * 0.5)) if digits else 3
    new_text = f"{new:,}" if "," in m.group() else str(new)
    return answer[:m.start()] + new_text + answer[m.end():], new_text


def detected(claims: list[dict], marker: str) -> bool:
    """Did the judge mark a claim containing the corrupted content as unsupported?"""
    marker_words = set(re.findall(r"\w+", marker.lower()))
    for c in claims:
        words = set(re.findall(r"\w+", c["claim"].lower()))
        hit = marker.lower() in c["claim"].lower() or (
            len(marker_words) > 3 and len(words & marker_words) / len(marker_words) >= 0.6)
        if hit and not c["supported_by_any"]:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = load_config()
    run_cfg = json.loads((args.run / "run_config.json").read_text(encoding="utf-8"))
    with (cfg.paths.index_dir / run_cfg["chunking"] / "chunks.jsonl").open(encoding="utf-8") as f:
        chunks = {c["chunk_id"]: Chunk(**c) for c in map(json.loads, f)}
    with cfg.paths.manifest.open(encoding="utf-8") as f:
        retrieved_at = {r["doc_id"]: r["retrieved_at"] for r in csv.DictReader(f)}
    with (args.run / "answers.jsonl").open(encoding="utf-8") as f:
        answers = [a for a in map(json.loads, f) if not a["refused"]]

    rng = random.Random(args.seed)
    sample = rng.sample(answers, min(args.n, len(answers)))
    judge = make_llm(cfg.judge)
    rows, cost = [], 0.0
    for a in sample:
        sources = build_sources_block([chunks[r["chunk_id"]] for r in a["retrieved"]],
                                      retrieved_at)
        cases = []
        num = corrupt_number(a["answer"], rng)
        if num:
            cases.append(("number", num[0], num[1]))
        addition = rng.choice(ADDITIONS)
        cases.append(("addition", a["answer"].rstrip() + " " + addition, addition))
        for kind, text, marker in cases:
            res = judge_faithfulness(judge, sources, text)
            cost += res.response.usage.cost_usd(cfg.pricing_usd_per_mtok.get(
                res.response.model)) or 0.0
            claims = (res.raw or {}).get("claims") or []
            ok = detected(claims, marker)
            rows.append({"id": a["id"], "kind": kind, "marker": marker, "detected": ok})
            print(f"{a['id']} {kind:8s} {'CAUGHT' if ok else 'missed'}  [{marker[:70]}]")

    out = args.run / "judge_perturbation.json"
    summary = {}
    for kind in ("number", "addition"):
        r = [x for x in rows if x["kind"] == kind]
        summary[kind] = {"n": len(r), "detected": sum(x["detected"] for x in r),
                         "recall": sum(x["detected"] for x in r) / len(r) if r else None}
    out.write_text(json.dumps({"judge": cfg.judge.model_dump(), "summary": summary,
                               "rows": rows, "cost_usd": cost}, indent=2),
                   encoding="utf-8", newline="\n")
    print(f"\n{json.dumps(summary)}\njudge cost ${cost:.3f} -> {out}")


if __name__ == "__main__":
    main()

"""Paired comparison of two retrieval configurations from one evaluation run.

Both configs answer the SAME questions, so their per-question scores are correlated.
Comparing two separate confidence intervals ignores that and is too conservative.
The paired bootstrap resamples questions and recomputes the difference each time.
If the 95% CI of the difference excludes 0, the gap is unlikely to be sampling noise.

    python scripts/compare_configs.py                       # latest run, default pairs
    python scripts/compare_configs.py --run eval/results/retrieval_XXXX --metric rr
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from visaguide.config import ROOT

DEFAULT_PAIRS = [
    # (config A, config B): reported as B - A
    (("fixed", "dense"), ("fixed", "hybrid")),
    (("structure", "dense"), ("structure", "hybrid")),
    (("fixed", "hybrid"), ("fixed", "hybrid_rerank")),
    (("structure", "hybrid"), ("structure", "hybrid_rerank")),
    (("fixed", "dense"), ("fixed", "hybrid_rerank")),
    (("structure", "dense"), ("structure", "hybrid_rerank")),
    (("fixed", "bm25"), ("fixed", "dense")),
    (("fixed", "dense"), ("structure", "dense")),
    (("fixed", "hybrid_rerank"), ("structure", "hybrid_rerank")),
]


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    diffs = (b[idx] - a[idx]).mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float((b - a).mean()), float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, help="results folder (default: latest retrieval run)")
    ap.add_argument("--metrics", nargs="+", default=["recall@5", "rr"])
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    run = args.run or sorted((ROOT / "eval" / "results").glob("retrieval_*"))[-1]
    scores: dict = defaultdict(dict)
    with (run / "per_question.jsonl").open(encoding="utf-8") as f:
        for r in map(json.loads, f):
            scores[(r["chunking"], r["mode"])][r["id"]] = r

    lines = [f"Paired bootstrap (10,000 resamples) - {run.name}\n",
             "| Comparison (B - A) | Metric | Δ | 95% CI | Significant? |",
             "|---|---|---|---|---|"]
    for a, b in DEFAULT_PAIRS:
        if a not in scores or b not in scores:
            continue
        ids = sorted(scores[a])
        for m in args.metrics:
            va = np.array([scores[a][i][m] for i in ids])
            vb = np.array([scores[b][i][m] for i in ids])
            d, lo, hi = paired_bootstrap(va, vb)
            sig = "yes" if lo > 0 or hi < 0 else "no"
            name = "MRR" if m == "rr" else m
            lines.append(f"| {'/'.join(b)} − {'/'.join(a)} | {name} | {d:+.3f} | "
                         f"[{lo:+.3f}, {hi:+.3f}] | {sig} |")
    out = "\n".join(lines) + "\n"
    (run / "paired_comparisons.md").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

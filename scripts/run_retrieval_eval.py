"""Retrieval evaluation: every chunking strategy x every retrieval mode.

    python scripts/run_retrieval_eval.py               # test split (the numbers we report)
    python scripts/run_retrieval_eval.py --split dev   # use dev for any tuning

Writes eval/results/retrieval_<timestamp>/ containing:
    table.md            the results table (paste into the README)
    metrics.json        the same numbers, with confidence intervals and run metadata
    per_question.jsonl  one row per (config, question): for error analysis
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from visaguide.chunking import STRATEGIES
from visaguide.config import ROOT, load_config
from visaguide.evaluation.anchor import normalise
from visaguide.evaluation.retrieval_metrics import recall_at_k, reciprocal_rank
from visaguide.retrieval.dense import SentenceTransformerEncoder
from visaguide.retrieval.retriever import MODES, Retriever

K_VALUES = (1, 3, 5, 10)
TESTSET = ROOT / "eval" / "testset.jsonl"


def bootstrap_ci(values: list[float], n_resamples: int = 2000, seed: int = 0):
    """95% percentile-bootstrap CI of the mean: resample questions with replacement.

    It shows how much the metric would move with a different sample of questions of
    the same kind. If two configs' intervals overlap heavily, don't claim one is better.
    """
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(values)
    means = arr[rng.integers(0, len(arr), (n_resamples, len(arr)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_questions(split: str, docs_by_id: dict) -> list[dict]:
    with TESTSET.open(encoding="utf-8") as f:
        qs = [json.loads(line) for line in f]
    # Guard: gold offsets must still point at the quoted text. If the corpus was
    # re-cleaned, offsets may have shifted, so re-run anchor_testset.py.
    for q in qs:
        for g in q["gold"]:
            actual = docs_by_id[g["doc_id"]]["text"][g["start"]:g["end"]]
            if normalise(actual) != normalise(g["quote"]):
                sys.exit(f"Gold passage of {q['id']} no longer matches the corpus. "
                         "Re-run scripts/anchor_testset.py.")
    qs = [q for q in qs if q["answerable"]]  # retrieval metrics need a gold passage
    return qs if split == "all" else [q for q in qs if q["split"] == split]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=ROOT, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "dev", "all"], default="test")
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--chunkings", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = load_config()
    with cfg.paths.processed.open(encoding="utf-8") as f:
        docs_by_id = {d["doc_id"]: d for d in map(json.loads, f)}
    questions = load_questions(args.split, docs_by_id)
    if not questions:
        sys.exit(f"No answerable questions in split '{args.split}'.")
    print(f"{len(questions)} answerable questions ({args.split} split)\n")

    encoder = SentenceTransformerEncoder(cfg.retrieval.embedding_model, cfg.retrieval.query_prefix)
    shared_reranker = None
    k_max = max(K_VALUES)
    rows, per_question = [], []

    for chunking in args.chunkings:
        retriever = Retriever.load(cfg.paths.index_dir / chunking, cfg.retrieval, encoder=encoder)
        retriever.reranker = shared_reranker
        for mode in args.modes:
            retriever.search("warm-up query", mode=mode, k=k_max)  # loads models, fills caches
            shared_reranker = retriever.reranker or shared_reranker
            recalls = {k: [] for k in K_VALUES}
            rrs, latencies = [], []
            for q in questions:
                t0 = time.perf_counter()
                hits = retriever.search(q["question"], mode=mode, k=k_max)
                latencies.append((time.perf_counter() - t0) * 1000)
                chunks = [h.chunk for h in hits]
                for k in K_VALUES:
                    recalls[k].append(recall_at_k(chunks, q["gold"], k))
                rrs.append(reciprocal_rank(chunks, q["gold"]))
                per_question.append({
                    "id": q["id"], "chunking": chunking, "mode": mode,
                    **{f"recall@{k}": recalls[k][-1] for k in K_VALUES},
                    "rr": rrs[-1], "latency_ms": round(latencies[-1], 1),
                    "top_chunks": [c.chunk_id for c in chunks],
                })
            row = {
                "chunking": chunking, "mode": mode, "n": len(questions),
                **{f"recall@{k}": float(np.mean(recalls[k])) for k in K_VALUES},
                "mrr": float(np.mean(rrs)),
                "recall@5_ci": bootstrap_ci(recalls[5]), "mrr_ci": bootstrap_ci(rrs),
                "latency_p50_ms": float(np.percentile(latencies, 50)),
                "latency_p95_ms": float(np.percentile(latencies, 95)),
            }
            rows.append(row)
            print(f"{chunking:9s} {mode:13s} R@5={row['recall@5']:.3f}  MRR={row['mrr']:.3f}  "
                  f"p50={row['latency_p50_ms']:.0f}ms")

    out = ROOT / "eval" / "results" / f"retrieval_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    table = render_table(rows, args.split)
    (out / "table.md").write_text(table, encoding="utf-8", newline="\n")
    meta = {
        "split": args.split, "n_questions": len(questions), "git_commit": git_commit(),
        "testset_sha256": hashlib.sha256(TESTSET.read_bytes()).hexdigest()[:12],
        "retrieval_cfg": cfg.retrieval.model_dump(), "chunking_cfg": cfg.chunking.model_dump(),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "metrics.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2),
                                      encoding="utf-8", newline="\n")
    with (out / "per_question.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in per_question:
            f.write(json.dumps(r) + "\n")
    print("\n" + table)
    print(f"Saved to {Path(out).relative_to(ROOT)}")


def render_table(rows: list[dict], split: str) -> str:
    lines = [
        f"Retrieval results - {split} split, n = {rows[0]['n']} answerable questions. "
        "Brackets: 95% bootstrap CI.\n",
        "| Chunking | Retriever | R@1 | R@3 | R@5 | R@10 | MRR | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lo5, hi5 = r["recall@5_ci"]
        lom, him = r["mrr_ci"]
        lines.append(
            f"| {r['chunking']} | {r['mode']} | {r['recall@1']:.3f} | {r['recall@3']:.3f} | "
            f"{r['recall@5']:.3f} [{lo5:.2f}-{hi5:.2f}] | {r['recall@10']:.3f} | "
            f"{r['mrr']:.3f} [{lom:.2f}-{him:.2f}] | {r['latency_p50_ms']:.0f} | "
            f"{r['latency_p95_ms']:.0f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

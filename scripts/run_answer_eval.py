"""End-to-end answer evaluation: generate answers, judge them, aggregate.

    python scripts/run_answer_eval.py --split dev --limit 3      # smoke test: measure cost
    python scripts/run_answer_eval.py --split test               # full run (costs money)
    python scripts/run_answer_eval.py --resume eval/results/answers_test_...   # continue

Stages write to disk as they go (answers.jsonl, judgments.jsonl), so a crash or a
rate limit never loses paid work: --resume skips everything already done. Judging
reads saved answers, so you can re-judge without paying to regenerate.

Metrics
  refusal      - unanswerable questions: strict refusal (coverage "none") and
                 gap flagged (coverage "none" or "partial")
                 answerable questions: false-refusal rate, partial-answer rate
  correctness  - judge vs reference: correct=1, partial=0.5, incorrect=0
                 (over all answerable questions; a refusal there scores 0)
  faithfulness - share of answer claims supported by the retrieved sources
  citation acc - share of claims supported by the source(s) they cite
  quote check  - share of citation quotes found verbatim in the cited chunk
  gold cite    - answered questions where a cited chunk overlaps a gold passage
                 (a lower bound: gold labels are known to be incomplete)
  latency      - p50/p95 of retrieval, generation and total time per query
  cost         - generation cost per query from API token usage x config prices
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from visaguide.chunking import Chunk
from visaguide.config import ROOT, load_config
from visaguide.evaluation.judge import (
    CORRECTNESS_SCORE,
    faithfulness_scores,
    judge_correctness,
    judge_faithfulness,
)
from visaguide.evaluation.retrieval_metrics import matches
from visaguide.generation.answer import RAGPipeline
from visaguide.generation.llm import make_llm
from visaguide.generation.prompts import build_sources_block

TESTSET = ROOT / "eval" / "testset.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=ROOT, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def mean(values: list) -> float:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "dev", "all"], default="test")
    ap.add_argument("--chunking", default=None)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--top-k", type=int, default=0, help="chunks given to the LLM "
                    "(default: retrieval.top_k from config)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions")
    ap.add_argument("--resume", type=Path, help="existing results folder to continue")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = load_config()
    chunking = args.chunking or cfg.retrieval.default_chunking
    mode = args.mode or cfg.retrieval.default_mode
    top_k = args.top_k or cfg.retrieval.top_k
    with TESTSET.open(encoding="utf-8") as f:
        questions = [q for q in map(json.loads, f)
                     if args.split == "all" or q["split"] == args.split]
    if args.limit:
        questions = questions[:args.limit]

    if args.resume:
        out = args.resume
    else:
        tag = f"_{args.tag}" if args.tag else ""
        out = ROOT / "eval" / "results" / (
            f"answers_{args.split}_{chunking}_{mode}{tag}_{datetime.now():%Y%m%d-%H%M%S}")
        out.mkdir(parents=True)
    (out / "run_config.json").write_text(json.dumps({
        "split": args.split, "chunking": chunking, "mode": mode, "limit": args.limit,
        "generation": cfg.generation.model_dump(), "judge": cfg.judge.model_dump(),
        "top_k": top_k, "git_commit": git_commit(),
        "testset_sha256": hashlib.sha256(TESTSET.read_bytes()).hexdigest()[:12],
    }, indent=2), encoding="utf-8", newline="\n")

    # ---------- stage 1: generate ----------
    ans_path = out / "answers.jsonl"
    done = {r["id"] for r in read_jsonl(ans_path)}
    todo = [q for q in questions if q["id"] not in done]
    if todo:
        pipe = RAGPipeline.from_config(cfg, chunking=chunking, mode=mode, top_k=top_k)
        for n, q in enumerate(todo, 1):
            res = pipe.answer(q["question"])
            append_jsonl(ans_path, {"id": q["id"], **res.to_dict()})
            print(f"[gen {n}/{len(todo)}] {q['id']} refused={res.refused} "
                  f"cost=${res.cost_usd or 0:.4f} {res.timings_ms['total'] / 1000:.1f}s"
                  f"{' ERROR ' + res.error if res.error else ''}")

    answers = {r["id"]: r for r in read_jsonl(ans_path)}
    qs = {q["id"]: q for q in questions}

    # ---------- stage 2: judge ----------
    jud_path = out / "judgments.jsonl"
    if not args.skip_judge:
        judged = {r["id"] for r in read_jsonl(jud_path)}
        with (cfg.paths.index_dir / chunking / "chunks.jsonl").open(encoding="utf-8") as f:
            chunks_by_id = {c["chunk_id"]: Chunk(**c) for c in map(json.loads, f)}
        with cfg.paths.manifest.open(encoding="utf-8") as f:
            retrieved_at = {r["doc_id"]: r["retrieved_at"] for r in csv.DictReader(f)}
        judge = None
        for n, qid in enumerate([i for i in answers if i in qs and i not in judged], 1):
            judge = judge or make_llm(cfg.judge)
            a, q = answers[qid], qs[qid]
            row = {"id": qid, "faithfulness": None, "correctness": None, "judge_cost_usd": 0.0}
            if not a["refused"]:
                chunks = [chunks_by_id[r["chunk_id"]] for r in a["retrieved"]]
                fj = judge_faithfulness(judge, build_sources_block(chunks, retrieved_at),
                                        a["answer"])
                row["faithfulness"] = fj.raw
                row["judge_cost_usd"] += fj.response.usage.cost_usd(
                    cfg.pricing_usd_per_mtok.get(fj.response.model)) or 0.0
            if q["answerable"] and not a["refused"]:
                cj = judge_correctness(judge, q["question"], q["reference_answer"], a["answer"])
                row["correctness"] = cj.raw
                row["judge_cost_usd"] += cj.response.usage.cost_usd(
                    cfg.pricing_usd_per_mtok.get(cj.response.model)) or 0.0
            append_jsonl(jud_path, row)
            print(f"[judge {n}] {qid} "
                  f"{(row['correctness'] or {}).get('verdict', '-')} ${row['judge_cost_usd']:.4f}")

    # ---------- stage 3: aggregate ----------
    judgments = {r["id"]: r for r in read_jsonl(jud_path)}
    rows = []
    for qid, q in qs.items():
        a, j = answers.get(qid), judgments.get(qid, {})
        if a is None:
            continue
        claims = ((j.get("faithfulness") or {}).get("claims")) or []
        faith, cite_acc = faithfulness_scores(claims) if not a["refused"] else (None, None)
        if q["answerable"]:
            verdict = "incorrect" if a["refused"] else (j.get("correctness") or {}).get("verdict")
        else:
            verdict = None
        cits = a["citations"]
        gold_cite = None
        if q["answerable"] and not a["refused"]:
            cited = {c["chunk_id"] for c in cits}
            spans = [SimpleNamespace(doc_id=r["doc_id"], start=r["start"], end=r["end"])
                     for r in a["retrieved"] if r["chunk_id"] in cited]
            gold_cite = float(any(matches(s, g) for s in spans for g in q["gold"]))
        # Runs before v2 only had `refused`; map it onto coverage for comparability.
        coverage = a.get("coverage") or ("none" if a["refused"] else "full")
        rows.append({
            "id": qid, "answerable": q["answerable"], "type": q["type"], "country": q["country"],
            "refused": a["refused"], "coverage": coverage, "verdict": verdict,
            "correctness": CORRECTNESS_SCORE.get(verdict) if verdict else None,
            "faithfulness": faith, "citation_accuracy": cite_acc,
            "quote_verified": (sum(c["quote_verified"] for c in cits) / len(cits)
                               if cits else None),
            "gold_cite": gold_cite, "n_citations": len(cits),
            "latency_total_ms": a["timings_ms"]["total"],
            "latency_retrieval_ms": a["timings_ms"]["retrieval"],
            "latency_generation_ms": a["timings_ms"]["generation"],
            "cost_usd": a["cost_usd"], "judge_cost_usd": j.get("judge_cost_usd"),
            "error": a.get("error", ""),
        })

    ans_rows = [r for r in rows if r["answerable"]]
    una_rows = [r for r in rows if not r["answerable"]]
    m = {
        "n_questions": len(rows), "n_answerable": len(ans_rows), "n_unanswerable": len(una_rows),
        "correctness": mean([r["correctness"] for r in ans_rows]),
        "share_correct": mean([r["verdict"] == "correct" for r in ans_rows]),
        "faithfulness": mean([r["faithfulness"] for r in rows]),
        "citation_accuracy": mean([r["citation_accuracy"] for r in rows]),
        "quote_verified": mean([r["quote_verified"] for r in rows]),
        "gold_cite_hit": mean([r["gold_cite"] for r in ans_rows]),
        # Unanswerable: did it decline outright (strict), or at least flag the missing
        # information (none or partial)? Near-misses are best answered "partial".
        "refusal_recall_unanswerable": mean([r["refused"] for r in una_rows]),
        "gap_flagged_unanswerable": mean([r["coverage"] != "full" for r in una_rows]),
        "false_refusal_rate_answerable": mean([r["refused"] for r in ans_rows]),
        "partial_rate_answerable": mean([r["coverage"] == "partial" for r in ans_rows]),
        "latency_total_p50_ms": pct([r["latency_total_ms"] for r in rows], 50),
        "latency_total_p95_ms": pct([r["latency_total_ms"] for r in rows], 95),
        "latency_generation_p50_ms": pct([r["latency_generation_ms"] for r in rows], 50),
        "latency_generation_p95_ms": pct([r["latency_generation_ms"] for r in rows], 95),
        "latency_retrieval_p50_ms": pct([r["latency_retrieval_ms"] for r in rows], 50),
        "cost_per_query_usd_mean": mean([r["cost_usd"] for r in rows]),
        "cost_per_query_usd_p95": pct([r["cost_usd"] for r in rows if r["cost_usd"]], 95),
        "judge_cost_total_usd": float(sum(r["judge_cost_usd"] or 0 for r in rows)),
        "generation_cost_total_usd": float(sum(r["cost_usd"] or 0 for r in rows)),
        "errors": sum(bool(r["error"]) for r in rows),
    }
    with (out / "per_question.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8", newline="\n")
    print(f"\nResults in {out.relative_to(ROOT)}")
    for k, v in m.items():
        print(f"  {k:32s} {v:.4f}" if isinstance(v, float) else f"  {k:32s} {v}")


if __name__ == "__main__":
    main()

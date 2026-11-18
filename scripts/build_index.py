"""Chunk the processed corpus and build the dense index, once per chunking strategy.

Usage:
    python scripts/build_index.py                     # both strategies
    python scripts/build_index.py --strategy structure

BM25 isn't saved: rebuilding it from chunks.jsonl takes about a second, so storing
it would only add a file that can go stale.
"""

import argparse
import json
import statistics
import time
from datetime import UTC, datetime

from visaguide.chunking import STRATEGIES, WORD, chunk_document
from visaguide.config import load_config
from visaguide.retrieval.dense import DenseRetriever, SentenceTransformerEncoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=[*STRATEGIES, "all"], default="all")
    args = ap.parse_args()

    cfg = load_config()
    with cfg.paths.processed.open(encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    by_id = {d["doc_id"]: d for d in docs}
    encoder = SentenceTransformerEncoder(cfg.retrieval.embedding_model,
                                         cfg.retrieval.query_prefix)

    for strategy in STRATEGIES if args.strategy == "all" else [args.strategy]:
        t0 = time.perf_counter()
        chunks = [c for d in docs for c in chunk_document(d, strategy, cfg.chunking)]
        for c in chunks:  # guard the invariant the evaluation depends on
            assert by_id[c.doc_id]["text"][c.start:c.end] == c.text

        out = cfg.paths.index_dir / strategy
        out.mkdir(parents=True, exist_ok=True)
        with (out / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for c in chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

        print(f"[{strategy}] embedding {len(chunks)} chunks ...")
        DenseRetriever.build(encoder, [c.embed_text() for c in chunks]).save(out / "dense.faiss")

        words = [len(WORD.findall(c.text)) for c in chunks]
        meta = {
            "strategy": strategy,
            "embedding_model": cfg.retrieval.embedding_model,
            "chunking": getattr(cfg.chunking, strategy).model_dump(),
            "n_docs": len(docs), "n_chunks": len(chunks),
            "words_mean": round(statistics.mean(words), 1),
            "words_median": statistics.median(words), "words_max": max(words),
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8",
                                       newline="\n")
        print(f"[{strategy}] {len(chunks)} chunks, words mean {meta['words_mean']} / "
              f"median {meta['words_median']} / max {meta['words_max']}, "
              f"{time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()

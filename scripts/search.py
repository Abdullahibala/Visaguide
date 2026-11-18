"""Interactive retrieval check from the command line.

    python scripts/search.py "How much money do I need in a blocked account?"
    python scripts/search.py "student residence permit documents" --mode bm25 --chunking fixed -k 3
"""

import argparse
import sys

from visaguide.config import load_config
from visaguide.retrieval.retriever import MODES, Retriever


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    ap.add_argument("--chunking", choices=["fixed", "structure"], default="structure")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    cfg = load_config()
    retriever = Retriever.load(cfg.paths.index_dir / args.chunking, cfg.retrieval)
    hits = retriever.search(args.query, mode=args.mode, k=args.k)

    print(f"\n{args.mode} / {args.chunking}   timings (ms): {retriever.last_timings_ms}\n")
    for h in hits:
        c = h.chunk
        snippet = " ".join(c.text.split())[:350]
        print(f"#{h.rank}  score={h.score:.4f}  {c.chunk_id}  [{c.country}]")
        print(f"    {c.title}  {('> ' + c.heading) if c.heading else ''}")
        print(f"    {c.url}")
        print(f"    {snippet}{'...' if len(c.text) > 350 else ''}\n")


if __name__ == "__main__":
    main()

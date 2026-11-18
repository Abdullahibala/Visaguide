"""Why did retrieval miss this question? Prints, per chunking strategy:
- which chunks contain each gold passage,
- the rank of those chunks under dense and BM25 (searching the whole index),
- the top-3 dense hits, i.e. what beat the gold chunk.

    python scripts/diagnose_question.py q016 q025
"""

import json
import sys

from visaguide.chunking import STRATEGIES
from visaguide.config import ROOT, load_config
from visaguide.evaluation.retrieval_metrics import matches
from visaguide.retrieval.dense import SentenceTransformerEncoder
from visaguide.retrieval.retriever import Retriever


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    with (ROOT / "eval" / "testset.jsonl").open(encoding="utf-8") as f:
        qs = {q["id"]: q for q in map(json.loads, f)}
    encoder = SentenceTransformerEncoder(cfg.retrieval.embedding_model, cfg.retrieval.query_prefix)

    for strategy in STRATEGIES:
        r = Retriever.load(cfg.paths.index_dir / strategy, cfg.retrieval, encoder=encoder)
        n = len(r.chunks)
        for qid in sys.argv[1:]:
            q = qs[qid]
            print(f"\n=== {qid} [{strategy}] {q['question']}")
            dense_hits = r.dense.search(q["question"], n)
            dense_rank = {i: rank for rank, (i, _) in enumerate(dense_hits, 1)}
            bm25_rank = {i: rank for rank, (i, _) in enumerate(r.bm25.search(q["question"], n), 1)}
            for g in q["gold"]:
                hits = [i for i, c in enumerate(r.chunks) if matches(c, g)]
                print(f"  gold[{g['group']}] {g['doc_id']}: {' '.join(g['quote'].split())[:90]}")
                for i in hits:
                    c = r.chunks[i]
                    print(f"     in {c.chunk_id} ({len(c.text.split())} words, heading: "
                          f"{c.heading[:60]!r}) -> dense rank {dense_rank.get(i, '-')}, "
                          f"bm25 rank {bm25_rank.get(i, '-')}")
                if not hits:
                    print("     NO chunk satisfies the 50% overlap rule (boundary problem)")
            top = sorted(dense_rank, key=dense_rank.get)[:3]
            for i in top:
                c = r.chunks[i]
                snippet = " ".join(c.text.split())[:100]
                print(f"  dense #{dense_rank[i]}: {c.title[:45]} | {snippet}")


if __name__ == "__main__":
    main()

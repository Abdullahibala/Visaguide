"""One entry point for the four retrieval configurations we compare.

    dense          FAISS over bge-small embeddings
    bm25           BM25 over the same chunk texts
    hybrid         RRF fusion of dense top-N and BM25 top-N
    hybrid_rerank  hybrid, then a cross-encoder re-scores the top rerank_top_n

The index for one chunking strategy is stored in data/index/<strategy>/:
    chunks.jsonl   chunk records (offsets, text, metadata)
    dense.faiss    embedding index
    meta.json      embedding model and chunking parameters used to build it
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from visaguide.chunking import Chunk
from visaguide.retrieval.bm25 import BM25Retriever
from visaguide.retrieval.dense import DenseRetriever, Encoder, SentenceTransformerEncoder
from visaguide.retrieval.hybrid import reciprocal_rank_fusion

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    rank: int
    score: float
    chunk: Chunk


@dataclass
class Retriever:
    chunks: list[Chunk]
    dense: DenseRetriever
    bm25: BM25Retriever
    retrieval_cfg: object  # RetrievalCfg
    reranker: object | None = None  # Reranker, loaded lazily (~90 MB model)
    last_timings_ms: dict = field(default_factory=dict)

    def __post_init__(self):
        self._texts = [c.embed_text() for c in self.chunks]

    @classmethod
    def load(cls, index_dir: Path, retrieval_cfg, encoder: Encoder | None = None) -> "Retriever":
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        if meta["embedding_model"] != retrieval_cfg.embedding_model:
            raise ValueError(
                f"Index built with {meta['embedding_model']} but config says "
                f"{retrieval_cfg.embedding_model}. Rebuild: python scripts/build_index.py"
            )
        with (index_dir / "chunks.jsonl").open(encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f]
        encoder = encoder or SentenceTransformerEncoder(
            retrieval_cfg.embedding_model, retrieval_cfg.query_prefix)
        dense = DenseRetriever.load(encoder, index_dir / "dense.faiss")
        bm25 = BM25Retriever([c.embed_text() for c in chunks])
        return cls(chunks, dense, bm25, retrieval_cfg)

    def _get_reranker(self):
        if self.reranker is None:
            from visaguide.retrieval.rerank import Reranker
            self.reranker = Reranker(self.retrieval_cfg.reranker_model)
        return self.reranker

    def search(self, query: str, mode: str = "hybrid_rerank", k: int = 5) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        cfg, t = self.retrieval_cfg, {}
        n = cfg.candidates

        def timed(name, fn):
            t0 = time.perf_counter()
            out = fn()
            t[name] = round((time.perf_counter() - t0) * 1000, 1)
            return out

        if mode == "dense":
            ranked = timed("dense", lambda: self.dense.search(query, k))
        elif mode == "bm25":
            ranked = timed("bm25", lambda: self.bm25.search(query, k))
        else:
            d = timed("dense", lambda: self.dense.search(query, n))
            b = timed("bm25", lambda: self.bm25.search(query, n))
            fused = timed("fusion", lambda: reciprocal_rank_fusion(
                [[i for i, _ in d], [i for i, _ in b]], k=cfg.rrf_k))
            if mode == "hybrid":
                ranked = fused[:k]
            else:
                ids = [i for i, _ in fused[:cfg.rerank_top_n]]
                reranker = self._get_reranker()
                ranked = timed("rerank", lambda: reranker.rerank(query, ids, self._texts, k))

        self.last_timings_ms = t
        return [Hit(rank=r, score=s, chunk=self.chunks[i])
                for r, (i, s) in enumerate(ranked, 1)]

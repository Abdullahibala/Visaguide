"""Dense retrieval: bi-encoder embeddings + exact FAISS search.

Why IndexFlatIP (exact, brute-force)? With a few thousand chunks, an exact search
over every vector takes about a millisecond. Approximate indexes (IVF, HNSW) only
pay off at hundreds of thousands of vectors and would add recall error, which would
mix into our retrieval comparison. With L2-normalised vectors, inner product equals
cosine similarity.
"""

from pathlib import Path
from typing import Protocol

import faiss
import numpy as np


class Encoder(Protocol):
    def encode_passages(self, texts: list[str]) -> np.ndarray: ...
    def encode_queries(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    def __init__(self, model_name: str, query_prefix: str = ""):
        from sentence_transformers import SentenceTransformer  # heavy import, only when used

        self.model = SentenceTransformer(model_name, device="cpu")
        self.query_prefix = query_prefix

    def _encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        emb = self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=show_progress)
        return emb.astype(np.float32)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, show_progress=len(texts) > 100)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])


class DenseRetriever:
    def __init__(self, encoder: Encoder, index: faiss.Index):
        self.encoder = encoder
        self.index = index

    @classmethod
    def build(cls, encoder: Encoder, texts: list[str]) -> "DenseRetriever":
        emb = encoder.encode_passages(texts)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        return cls(encoder, index)

    def save(self, path: Path) -> None:
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, encoder: Encoder, path: Path) -> "DenseRetriever":
        return cls(encoder, faiss.read_index(str(path)))

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self.encoder.encode_queries([query])
        scores, ids = self.index.search(q, min(k, self.index.ntotal))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i >= 0]

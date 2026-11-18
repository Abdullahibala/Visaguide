"""Cross-encoder reranking.

A bi-encoder (dense retrieval) embeds the query and the chunk separately, so it can
pre-compute chunk vectors and search millions of them fast, but the two texts never
"see" each other. A cross-encoder reads [query, chunk] together through every
attention layer, which is far more accurate but costs one model forward pass per
pair. Hence the two-stage design: cheap retrievers find about 30 candidates, and
the expensive model orders them.
"""

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name, device="cpu", max_length=512)

    def rerank(self, query: str, ids: list[int], texts: list[str],
               k: int) -> list[tuple[int, float]]:
        if not ids:
            return []
        scores = self.model.predict([(query, texts[i]) for i in ids], batch_size=32)
        ranked = sorted(zip(ids, scores, strict=True), key=lambda p: -p[1])
        return [(int(i), float(s)) for i, s in ranked[:k]]

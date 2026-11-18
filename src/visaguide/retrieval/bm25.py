"""Sparse lexical retrieval with BM25.

Why keep BM25 next to embeddings? Visa questions are full of exact terms that small
embedding models blur: "§16b", "Sperrkonto", "e-ikamet", "APS", "11,904". BM25
rewards exact rare-term matches (through IDF), and embeddings reward paraphrases
("money in a locked account" ≈ "blocked account"). Their errors differ, which is
why fusing them helps.
"""

import re

import numpy as np
from rank_bm25 import BM25Okapi

TOKEN = re.compile(r"\w+", re.UNICODE)
# A small stopword list: common words carry almost no IDF signal but inflate scores
# for long chunks. We don't stem, so "permit" vs "permits" stay distinct. That's a
# known simplification to discuss.
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its may "
    "must my of on or should that the their there this to was we what when where which "
    "who will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Retriever:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        # k1: how quickly repeated terms saturate; b: how much long chunks are penalised.
        # These are the standard defaults; we don't tune them, to avoid overfitting.
        self.bm25 = BM25Okapi([tokenize(t) for t in texts], k1=k1, b=b)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

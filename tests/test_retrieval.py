import numpy as np
import pytest

from visaguide.retrieval.bm25 import BM25Retriever, tokenize
from visaguide.retrieval.dense import DenseRetriever
from visaguide.retrieval.hybrid import reciprocal_rank_fusion

CORPUS = [
    "The blocked account (Sperrkonto) must hold the required amount for one year.",
    "International students in Türkiye apply for a student residence permit via e-ikamet.",
    "Health insurance is mandatory for enrolment at a German university.",
    "Students may work 140 full days or 280 half days per year.",
]


# ---------------- RRF (you implement this one) ----------------

def test_rrf_item_ranked_well_by_both_wins():
    # 1: 1/61 + 1/62 ;  3: 1/63 + 1/61 ;  2: 1/62 ;  4: 1/63
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]], k=60)
    assert [i for i, _ in fused] == [1, 3, 2, 4]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_includes_items_from_only_one_list():
    fused = dict(reciprocal_rank_fusion([[7], [8, 9]], k=60))
    assert set(fused) == {7, 8, 9}
    assert fused[9] == pytest.approx(1 / 62)


def test_rrf_ties_broken_by_id_for_determinism():
    fused = reciprocal_rank_fusion([[5], [2]], k=60)  # both 1/61
    assert [i for i, _ in fused] == [2, 5]


def test_rrf_smaller_k_rewards_top_rank_more():
    # Item 10 is ranked 1st by one retriever only; item 20 is ranked 4th by both.
    #   k=60: 10 -> 1/61 = 0.0164   20 -> 2/64 = 0.0313   (agreement wins)
    #   k=1:  10 -> 1/2  = 0.5      20 -> 2/5  = 0.4      (top rank wins)
    rankings = [[10, 1, 2, 20], [30, 3, 4, 20]]
    big_k = dict(reciprocal_rank_fusion(rankings, k=60))
    small_k = dict(reciprocal_rank_fusion(rankings, k=1))
    assert big_k[20] > big_k[10]
    assert small_k[10] > small_k[20]


def test_rrf_weights_downweight_a_ranking():
    # With equal weights 1 and 2 would tie (1/61 + 1/62 each). Down-weighting the
    # second ranking to 0.1 lets the first ranking's order decide.
    fused = reciprocal_rank_fusion([[1, 2], [2, 1]], k=60, weights=[1.0, 0.1])
    assert [i for i, _ in fused] == [1, 2]
    assert fused[0][1] == pytest.approx(1 / 61 + 0.1 / 62)


def test_rrf_default_weights_equal_plain_rrf():
    r = [[1, 2, 3], [3, 1, 4]]
    assert reciprocal_rank_fusion(r) == reciprocal_rank_fusion(r, weights=[1.0, 1.0])


def test_rrf_empty_input():
    assert reciprocal_rank_fusion([[], []]) == []


# ---------------- BM25 ----------------

def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Blocked Account §16b") == ["blocked", "account", "16b"]


def test_bm25_exact_rare_term():
    bm25 = BM25Retriever(CORPUS)
    assert bm25.search("e-ikamet", k=2)[0][0] == 1
    assert bm25.search("Sperrkonto", k=2)[0][0] == 0


def test_bm25_returns_nothing_for_unknown_terms():
    assert BM25Retriever(CORPUS).search("zzzz qqqq", k=3) == []


# ---------------- Dense (with a fake encoder, so no model download) ----------------

class FakeEncoder:
    """Bag-of-words hashing encoder: deterministic and enough to test the plumbing."""

    def _vec(self, text):
        v = np.zeros(64, dtype=np.float32)
        for t in tokenize(text):
            v[hash(t) % 64] += 1
        return v / (np.linalg.norm(v) or 1)

    def encode_passages(self, texts):
        return np.stack([self._vec(t) for t in texts])

    encode_queries = encode_passages


def test_dense_search_and_save_load(tmp_path):
    dense = DenseRetriever.build(FakeEncoder(), CORPUS)
    assert dense.search("health insurance enrolment", k=1)[0][0] == 2
    dense.save(tmp_path / "x.faiss")
    again = DenseRetriever.load(FakeEncoder(), tmp_path / "x.faiss")
    assert again.search("work days per year", k=1)[0][0] == 3


def test_dense_k_larger_than_corpus():
    assert len(DenseRetriever.build(FakeEncoder(), CORPUS).search("permit", k=50)) == 4


# ---------------- Real models (slow: downloads ~200 MB on first run) ----------------

@pytest.mark.slow
def test_real_embeddings_match_paraphrase():
    from visaguide.retrieval.dense import SentenceTransformerEncoder
    enc = SentenceTransformerEncoder("BAAI/bge-small-en-v1.5",
                                     "Represent this sentence for searching relevant passages: ")
    dense = DenseRetriever.build(enc, CORPUS)
    # no word overlap with "blocked account"/"Sperrkonto": semantics only
    assert dense.search("how much money must I deposit to prove I can pay my living costs",
                        k=1)[0][0] == 0


@pytest.mark.slow
def test_reranker_prefers_relevant_passage():
    from visaguide.retrieval.rerank import Reranker
    rr = Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranked = rr.rerank("How many days can international students work?", [0, 1, 2, 3], CORPUS, k=1)
    assert ranked[0][0] == 3

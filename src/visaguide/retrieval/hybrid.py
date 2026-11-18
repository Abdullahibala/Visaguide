"""Hybrid retrieval: fuse the dense and BM25 rankings with Reciprocal Rank Fusion.

>>> YOUR TURN: implement reciprocal_rank_fusion() below (about 10 lines). <<<
The tests in tests/test_retrieval.py::test_rrf_* define the expected behaviour.
Run:  .venv\\Scripts\\python -m pytest tests/test_retrieval.py -k rrf -v

Background (be ready to explain this in an interview):
  RRF score(d) = sum over rankings r that contain d of  1 / (k + rank_r(d))
  with rank starting at 1 and k = 60.

  Why ranks and not scores? BM25 scores are unbounded (0..30+) and depend on the
  query, while cosine scores sit in a narrow band (0.5..0.9). Adding them needs
  normalisation, and min-max normalisation is sensitive to outliers. RRF uses only
  positions, so it needs no calibration and no tuned weights.
  Why k = 60? It flattens the curve: rank 1 vs rank 2 is 1/61 vs 1/62, so a
  document ranked well by BOTH retrievers beats one ranked first by only one.
  Smaller k gives more weight to the top positions.
"""


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Fuse several rankings of item ids into one.

    Args:
        rankings: each inner list holds item ids, best first (e.g. chunk indices
                  from the dense retriever and from BM25).
        k: the RRF constant.

    Returns:
        (item_id, rrf_score) pairs sorted by score, highest first. Every item that
        appears in at least one ranking is included. Break ties by item_id
        (ascending), so the output is deterministic.
    """
    raise NotImplementedError("Implement RRF here - see the module docstring.")

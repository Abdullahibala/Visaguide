"""Retrieval metrics: Recall@k and reciprocal rank (averaged over questions = MRR).

Definitions used throughout the project:

- A gold passage is a character range (doc_id, start, end) that answers the question.
- Gold passages carry a `group`. Passages in the same group are ALTERNATIVES (the
  same fact stated by two sources); different groups are DIFFERENT facts the answer
  needs. A group is "found" if any one of its passages is retrieved.
- A retrieved chunk is relevant to a gold passage if they are in the same document
  and their ranges share at least half of the shorter of the two ranges (see
  `matches`). The 50% rule is tolerant of chunk boundaries cutting through a
  passage (common for fixed-size chunks), without giving credit for a chunk that
  only touches the passage's edge.

>>> YOUR TURN: implement recall_at_k() and reciprocal_rank() at the bottom. <<<
Tests: .venv\\Scripts\\python -m pytest tests/test_metrics.py -v
"""

from typing import Protocol


class Span(Protocol):
    doc_id: str
    start: int
    end: int


def overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Number of characters shared by [a_start, a_end) and [b_start, b_end)."""
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def matches(chunk: Span, gold: dict, min_frac: float = 0.5) -> bool:
    if chunk.doc_id != gold["doc_id"]:
        return False
    shared = overlap(chunk.start, chunk.end, gold["start"], gold["end"])
    shorter = min(chunk.end - chunk.start, gold["end"] - gold["start"])
    return shorter > 0 and shared >= min_frac * shorter


def groups_found(chunk: Span, gold: list[dict]) -> set[str]:
    """The gold groups this one chunk satisfies (usually empty or one group)."""
    return {g["group"] for g in gold if matches(chunk, g)}


def all_groups(gold: list[dict]) -> set[str]:
    return {g["group"] for g in gold}


# ----------------------------------------------------------------------------------
# YOUR TURN. Use groups_found() and all_groups() above; don't re-implement overlap.
# ----------------------------------------------------------------------------------

def recall_at_k(retrieved: list[Span], gold: list[dict], k: int) -> float:
    """Fraction of gold groups found by at least one of the top-k retrieved chunks.

    `retrieved` is ordered best first. Example: 2 groups, and the top 5 chunks
    satisfy one of them -> 0.5. Return 0.0 if gold is empty (that shouldn't happen
    for answerable questions, but don't divide by zero).
    """
    raise NotImplementedError


def reciprocal_rank(retrieved: list[Span], gold: list[dict]) -> float:
    """1 / (rank of the first retrieved chunk that satisfies ANY gold group).

    Ranks start at 1. Return 0.0 if no retrieved chunk is relevant. The mean of
    this over all questions is MRR.
    """
    raise NotImplementedError

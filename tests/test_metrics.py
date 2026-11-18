from dataclasses import dataclass

import pytest

from visaguide.evaluation.anchor import find_quote, normalise
from visaguide.evaluation.retrieval_metrics import (
    groups_found,
    matches,
    recall_at_k,
    reciprocal_rank,
)


@dataclass
class C:  # minimal stand-in for a retrieved Chunk
    doc_id: str
    start: int
    end: int


GOLD = [
    {"doc_id": "d1", "start": 100, "end": 200, "group": "a"},  # fact A, source 1
    {"doc_id": "d2", "start": 0, "end": 50, "group": "a"},     # fact A, source 2 (alternative)
    {"doc_id": "d1", "start": 500, "end": 600, "group": "b"},  # fact B
]
HIT_A = C("d1", 90, 210)    # covers gold a
HIT_A2 = C("d2", 0, 300)    # covers the alternative for a
HIT_B = C("d1", 480, 700)   # covers gold b
MISS = C("d3", 100, 200)    # wrong document


# ---------------- anchoring ----------------

def test_find_quote_ignores_whitespace_markdown_and_case():
    doc = "Intro.\n\n**Proof of finances**\n\nA blocked   account\nis REQUIRED."
    start, end, n = find_quote(doc, "proof of finances a blocked account is required")
    assert doc[start:end] == "Proof of finances**\n\nA blocked   account\nis REQUIRED"
    assert n == 1


def test_find_quote_turkish_dotted_capital_i():
    doc = "Başvuru İkamet izni için yapılır."
    start, end, _ = find_quote(doc, "ikamet izni")
    assert doc[start:end] == "İkamet izni"


def test_find_quote_missing_returns_none():
    assert find_quote("some text", "not there at all") is None


def test_normalise_collapses_and_strips():
    assert normalise("  # Title\n\n- **Item**  ") == "title - item"


# ---------------- relevance rule (provided) ----------------

def test_matches_requires_same_doc_and_half_overlap():
    g = GOLD[0]  # d1 100-200
    assert matches(C("d1", 150, 400), g)       # shares 50 of the 100-char gold: exactly half
    assert not matches(C("d1", 160, 400), g)   # shares 40: not enough
    assert matches(C("d1", 120, 140), g)       # chunk fully inside gold
    assert not matches(C("d2", 100, 200), g)   # same offsets, different document


def test_groups_found():
    assert groups_found(HIT_A, GOLD) == {"a"}
    assert groups_found(MISS, GOLD) == set()


# ---------------- YOUR TURN: recall_at_k ----------------

def test_recall_all_groups_found():
    assert recall_at_k([HIT_A, HIT_B], GOLD, k=2) == 1.0


def test_recall_counts_groups_not_passages():
    # Two chunks satisfying the same group "a" still count as one of two groups.
    assert recall_at_k([HIT_A, HIT_A2, MISS], GOLD, k=3) == 0.5


def test_recall_only_looks_at_top_k():
    assert recall_at_k([MISS, MISS, HIT_A, HIT_B], GOLD, k=2) == 0.0
    assert recall_at_k([MISS, MISS, HIT_A, HIT_B], GOLD, k=3) == 0.5


def test_recall_empty_inputs():
    assert recall_at_k([], GOLD, k=5) == 0.0
    assert recall_at_k([HIT_A], [], k=5) == 0.0


# ---------------- YOUR TURN: reciprocal_rank ----------------

def test_rr_first_relevant_rank():
    assert reciprocal_rank([HIT_B, MISS], GOLD) == 1.0
    assert reciprocal_rank([MISS, MISS, HIT_A], GOLD) == pytest.approx(1 / 3)


def test_rr_any_group_counts():
    assert reciprocal_rank([MISS, HIT_A2], GOLD) == 0.5


def test_rr_no_relevant():
    assert reciprocal_rank([MISS, MISS], GOLD) == 0.0
    assert reciprocal_rank([], GOLD) == 0.0

"""Generation pipeline tests with a fake LLM: no API key, no cost, deterministic."""

import pytest

from visaguide.chunking import Chunk
from visaguide.config import Price
from visaguide.evaluation.agreement import cohen_kappa, raw_agreement
from visaguide.evaluation.judge import faithfulness_scores
from visaguide.generation.answer import RAGPipeline, verify_citations
from visaguide.generation.llm import LLMResponse, Usage
from visaguide.generation.prompts import build_user_prompt
from visaguide.retrieval.retriever import Hit


def chunk(i, text, doc="d1"):
    return Chunk(chunk_id=f"{doc}#s{i:04d}", doc_id=doc, start=0, end=len(text), text=text,
                 heading="Proof of finances", title="Student visa", url=f"https://x/{doc}",
                 country="germany")


CHUNKS = [chunk(0, "You need 11,904 euros in a blocked account."),
          chunk(1, "Health insurance is mandatory for enrolment.", doc="d2")]


class FakeRetriever:
    last_timings_ms = {"dense": 1.0}

    def search(self, q, mode, k):
        return [Hit(rank=i + 1, score=1.0, chunk=c) for i, c in enumerate(CHUNKS[:k])]


class FakeLLM:
    def __init__(self, data, model="claude-opus-5-5", stop_reason="end_turn"):
        self.data, self.model, self.stop_reason = data, model, stop_reason
        self.calls = []

    def generate_json(self, system, user, schema):
        self.calls.append((system, user))
        return LLMResponse(data=self.data, model=self.model, stop_reason=self.stop_reason,
                           usage=Usage(input_tokens=1000, output_tokens=200), latency_ms=5)


class Cfg:  # the parts of Config the pipeline uses
    class retrieval:
        default_mode, top_k = "dense", 5
    class paths:  # noqa: E301
        manifest = None
    pricing_usd_per_mtok = {"claude-opus-5-5": Price(input=4.0, output=20.0)}


@pytest.fixture
def pipeline(tmp_path):
    m = tmp_path / "manifest.csv"
    m.write_text("doc_id,retrieved_at\nd1,2026-11-18T10:00:00\nd2,2026-11-17T10:00:00\n",
                 encoding="utf-8")
    Cfg.paths.manifest = m
    return lambda llm: RAGPipeline(Cfg, FakeRetriever(), llm=llm)


def test_prompt_has_numbered_sources_dates_and_question_last():
    p = build_user_prompt("How much?", CHUNKS, {"d1": "2026-11-18T10:00:00"})
    assert '<source id="S1"' in p and '<source id="S2"' in p
    assert 'retrieved="2026-11-18"' in p and 'retrieved="unknown"' in p
    assert p.rstrip().endswith("Question: How much?")


def test_answer_maps_citations_and_verifies_quotes(pipeline):
    llm = FakeLLM({"answer": "You need €11,904 [S1].", "coverage": "full", "citations": [
        {"source": "S1", "quote": "You need 11,904 euros in a blocked account."},
        {"source": "S2", "quote": "Insurance is optional."},   # invented quote
        {"source": "S9", "quote": "anything"},                  # source that doesn't exist
    ]})
    res = pipeline(llm).answer("How much money?")
    assert [c.quote_verified for c in res.citations] == [True, False, False]
    assert res.citations[0].chunk_id == "d1#s0000"
    assert res.citations[2].chunk_id == ""
    assert res.cost_usd == pytest.approx((1000 * 4 + 200 * 20) / 1e6)
    assert "How much money?" in llm.calls[0][1]


def test_model_refusal_fails_closed(pipeline):
    res = pipeline(FakeLLM(None, stop_reason="refusal")).answer("q")
    assert res.refused and res.citations == [] and "couldn't" in res.answer


def test_unpriced_model_gives_unknown_cost_not_zero(pipeline):
    res = pipeline(FakeLLM({"answer": "x", "coverage": "none", "citations": []},
                           model="other-model")).answer("q")
    assert res.cost_usd is None


@pytest.mark.parametrize("coverage,refused", [("full", False), ("partial", False),
                                              ("none", True), ("bogus", True)])
def test_coverage_maps_to_refused(pipeline, coverage, refused):
    res = pipeline(FakeLLM({"answer": "x", "coverage": coverage, "citations": []})).answer("q")
    assert res.refused is refused
    assert res.coverage == (coverage if coverage != "bogus" else "none")


def test_verify_citations_tolerates_brackets_and_markdown():
    c = [chunk(0, "**Proof of finances:** a blocked   account.")]
    out = verify_citations([{"source": "[S1]", "quote": "proof of finances: a blocked account"}], c)
    assert out[0].quote_verified


def test_faithfulness_scores():
    claims = [{"supported_by_any": True, "supported_by_cited": True},
              {"supported_by_any": True, "supported_by_cited": False},
              {"supported_by_any": False, "supported_by_cited": False}]
    assert faithfulness_scores(claims) == (pytest.approx(2 / 3), pytest.approx(1 / 3))
    assert faithfulness_scores([]) == (None, None)


def test_kappa_perfect_chance_and_known_value():
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    # Textbook example: 20 items, 15 agreements (p_o = .75), marginals 50/50 vs 50/50
    a = ["y"] * 10 + ["n"] * 10
    b = ["y"] * 8 + ["n"] * 2 + ["y"] * 3 + ["n"] * 7
    assert raw_agreement(a, b) == 0.75
    assert cohen_kappa(a, b) == pytest.approx((0.75 - 0.5) / 0.5)

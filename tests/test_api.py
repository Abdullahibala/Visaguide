"""API tests with a fake pipeline: no models, no API key, no network, fast."""

import io
import json

import pytest
from fastapi.testclient import TestClient

from visaguide.api.main import client_ip, create_app
from visaguide.api.querylog import QueryLogger, hash_ip
from visaguide.api.ratelimit import DailyBudget, RateLimiter
from visaguide.config import load_config
from visaguide.generation.answer import AnswerResult, Citation
from visaguide.generation.llm import Usage


class FakePipeline:
    mode, top_k = "dense_rerank", 8
    retrieved_at = {"d1": "2026-11-18T10:00:00+00:00"}

    class retriever:
        chunks = [object()] * 3

    def __init__(self, cost=0.03, fail=False):
        self.cost, self.fail, self.questions = cost, fail, []

    def answer(self, question):
        self.questions.append(question)
        if self.fail:
            raise TimeoutError("upstream timed out")
        return AnswerResult(
            question=question, answer="You need €11,904 [S1].", refused=False, coverage="full",
            citations=[
                Citation("S1", "d1#s0001", "https://x/d1", "Student visa", "11,904 euros", True),
                Citation("S9", "", "", "", "made up", False),     # unknown id: must be dropped
            ],
            retrieved=[{"chunk_id": "d1#s0001", "doc_id": "d1", "start": 0, "end": 10,
                        "title": "Student visa", "url": "https://x/d1", "score": 0.9}],
            model="claude-opus-5-5", stop_reason="end_turn",
            usage=Usage(input_tokens=1000, output_tokens=100), cost_usd=self.cost,
            timings_ms={"retrieval": 5.0, "generation": 50.0, "total": 55.0})


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def make_client(tmp_path):
    def make(pipeline=None, per_minute=5, per_day=40, budget_usd=2.0, clock=None):
        cfg = load_config()
        stream = io.StringIO()
        app = create_app(
            cfg=cfg, pipeline=pipeline or FakePipeline(),
            limiter=RateLimiter(per_minute, per_day, clock=clock or Clock()),
            budget=DailyBudget(budget_usd), logger=QueryLogger(tmp_path / "q.jsonl", stream=stream))
        client = TestClient(app)
        client.__enter__()  # runs the lifespan (startup)
        client.log_stream, client.log_file = stream, tmp_path / "q.jsonl"
        return client
    return make


def ask(client, q="How much money do I need?", ip="203.0.113.7"):
    return client.post("/ask", json={"question": q}, headers={"X-Forwarded-For": ip})


def test_health(make_client):
    r = make_client().get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["index_chunks"] == 3
    assert body["retrieval"].endswith("dense_rerank/top8")
    assert body["budget"]["limit_usd"] == 2.0


def test_ask_returns_answer_and_only_known_citations(make_client):
    r = ask(make_client())
    assert r.status_code == 200
    body = r.json()
    assert body["coverage"] == "full" and "11,904" in body["answer"]
    assert [c["source"] for c in body["citations"]] == ["S1"]
    assert body["citations"][0]["retrieved_at"] == "2026-11-18"
    assert "not legal advice" in body["disclaimer"]


@pytest.mark.parametrize("question", ["", "  ", "hi", "x" * 501])
def test_ask_rejects_invalid_questions(make_client, question):
    assert ask(make_client(), q=question).status_code == 422


def test_question_whitespace_is_normalised(make_client):
    pipe = FakePipeline()
    ask(make_client(pipeline=pipe), q="  How   much\n money?  ")
    assert pipe.questions == ["How much money?"]


def test_rate_limit_per_minute_then_recovers(make_client):
    clock = Clock()
    client = make_client(per_minute=2, clock=clock)
    assert ask(client).status_code == 200
    assert ask(client).status_code == 200
    r = ask(client)
    assert r.status_code == 429 and int(r.headers["Retry-After"]) > 0
    assert ask(client, ip="198.51.100.1").status_code == 200   # other clients unaffected
    clock.t += 61
    assert ask(client).status_code == 200                      # window has passed


def test_rate_limit_per_day(make_client):
    clock = Clock()
    client = make_client(per_minute=100, per_day=3, clock=clock)
    for _ in range(3):
        assert ask(client).status_code == 200
        clock.t += 120
    assert ask(client).status_code == 429


def test_daily_budget_cap_stops_spending(make_client):
    client = make_client(pipeline=FakePipeline(cost=0.03), budget_usd=0.05)
    assert ask(client, ip="1.1.1.1").status_code == 200   # spent 0.03
    assert ask(client, ip="2.2.2.2").status_code == 200   # spent 0.06 >= 0.05
    r = ask(client, ip="3.3.3.3")                         # new IP doesn't help
    assert r.status_code == 503 and "budget" in r.json()["detail"]


def test_budget_resets_on_a_new_day():
    day = {"d": "2026-11-19"}
    b = DailyBudget(0.05, today=lambda: day["d"])
    b.add(0.06)
    assert not b.available()
    day["d"] = "2026-11-20"
    assert b.available() and b.status()["spent_usd"] == 0.0


def test_upstream_failure_returns_502_and_is_logged(make_client):
    client = make_client(pipeline=FakePipeline(fail=True))
    r = ask(client)
    assert r.status_code == 502 and "temporarily unavailable" in r.json()["detail"]
    rec = json.loads(client.log_file.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["status"] == 502 and "TimeoutError" in rec["error"]


def test_query_log_has_retrieval_latency_cost_and_no_raw_ip(make_client):
    client = make_client()
    ask(client, ip="203.0.113.7")
    rec = json.loads(client.log_file.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["status"] == 200 and rec["cost_usd"] == 0.03
    assert rec["retrieved"][0]["chunk_id"] == "d1#s0001"
    assert rec["timings_ms"]["generation"] == 50.0 and "endpoint_total" in rec["timings_ms"]
    assert rec["usage"]["input_tokens"] == 1000
    assert "203.0.113.7" not in client.log_file.read_text(encoding="utf-8")
    assert "203.0.113.7" not in client.log_stream.getvalue()   # stdout copy too
    assert rec["client"] == hash_ip("203.0.113.7", "dev-only-salt-change-me")


def test_client_ip_uses_rightmost_forwarded_for():
    class Req:
        headers = {"x-forwarded-for": "6.6.6.6, 203.0.113.7"}   # first value is client-forgeable
        client = type("C", (), {"host": "10.0.0.1"})
    assert client_ip(Req, trust_forwarded_for=True) == "203.0.113.7"
    assert client_ip(Req, trust_forwarded_for=False) == "10.0.0.1"


def test_index_page_served(make_client):
    r = make_client().get("/")
    assert r.status_code == 200 and "VisaGuide" in r.text and "textContent" in r.text

"""FastAPI app: POST /ask, GET /health, and the web UI at /.

Run locally:   uvicorn visaguide.api.main:app --port 8080
Interactive API docs (generated from the Pydantic models): http://localhost:8080/docs

create_app() takes its collaborators (pipeline, limiter, budget, logger) as
arguments, so the tests can inject fakes: no models, no API key, no network.
"""

import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from visaguide.api.querylog import QueryLogger, hash_ip
from visaguide.api.ratelimit import DailyBudget, RateLimiter
from visaguide.config import ROOT, Config, Secrets, load_config

STATIC = Path(__file__).parent / "static"
DISCLAIMER = ("VisaGuide is not legal advice. Answers come only from official public pages "
              "retrieved on the date shown; rules change, so confirm with the responsible "
              "authority before you act.")


class CitationOut(BaseModel):
    source: str
    title: str
    url: str
    quote: str
    quote_verified: bool = Field(description="quote found verbatim in the cited source")
    retrieved_at: str


class AskResponse(BaseModel):
    request_id: str
    answer: str
    coverage: str = Field(description="full | partial | none")
    citations: list[CitationOut]
    latency_ms: float
    disclaimer: str = DISCLAIMER


def client_ip(request: Request, trust_forwarded_for: bool) -> str:
    """Behind Cloud Run's front end the TCP peer is Google's proxy, not the user.
    The proxy appends the real client IP to X-Forwarded-For. A client can put
    anything at the *start* of that header, so we take the rightmost value: the
    one added by the trusted proxy, which the client can't forge."""
    if trust_forwarded_for:
        xff = request.headers.get("x-forwarded-for", "")
        if xff.strip():
            return xff.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def create_app(cfg: Config | None = None, pipeline=None, limiter: RateLimiter | None = None,
               budget: DailyBudget | None = None, logger: QueryLogger | None = None) -> FastAPI:
    cfg = cfg or load_config()
    salt = Secrets().ip_hash_salt
    limiter = limiter or RateLimiter(cfg.api.rate_limit_per_minute, cfg.api.rate_limit_per_day)
    budget = budget or DailyBudget(cfg.api.daily_budget_usd)
    log_path = cfg.api.log_path if cfg.api.log_path.is_absolute() else ROOT / cfg.api.log_path
    logger = logger or QueryLogger(log_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load the embedding model, reranker and index ONCE, at startup.
        if pipeline is None:
            from visaguide.generation.answer import RAGPipeline
            app.state.pipeline = RAGPipeline.from_config(cfg)
        else:
            app.state.pipeline = pipeline
        yield

    app = FastAPI(title="VisaGuide", version=os.environ.get("GIT_COMMIT", "dev"),
                  description="Cited answers about study visas and residence permits "
                              "(Germany, Türkiye) from official sources.",
                  lifespan=lifespan)

    class AskRequest(BaseModel):
        question: str = Field(min_length=3, max_length=cfg.api.max_question_chars,
                              examples=["How much money do I need in my blocked account?"])

        @field_validator("question")
        @classmethod
        def strip(cls, v: str) -> str:
            v = " ".join(v.split())
            if len(v) < 3:
                raise ValueError("question is too short")
            return v

    @app.get("/health")
    def health() -> dict:
        p = app.state.pipeline
        return {
            "status": "ok",
            "version": app.version,
            "generation_model": cfg.generation.model,
            "retrieval": f"{cfg.retrieval.default_chunking}/{getattr(p, 'mode', '?')}"
                         f"/top{getattr(p, 'top_k', '?')}",
            "index_chunks": len(getattr(getattr(p, "retriever", None), "chunks", []) or []),
            "budget": budget.status(),
        }

    # A plain `def` (not async): the pipeline is blocking (torch + HTTP), so FastAPI
    # runs it in a thread pool and the event loop stays free for other requests.
    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request) -> AskResponse:
        rid = uuid.uuid4().hex[:12]
        ip = client_ip(request, cfg.api.trust_forwarded_for)
        base = {"event": "ask", "request_id": rid, "client": hash_ip(ip, salt),
                "question": body.question}

        allowed, retry_after = limiter.check(ip)
        if not allowed:
            logger.log({**base, "status": 429})
            raise HTTPException(429, "Too many questions from you right now. "
                                     "Please try again later.",
                                headers={"Retry-After": str(retry_after)})
        if not budget.available():
            logger.log({**base, "status": 503, "budget": budget.status()})
            raise HTTPException(503, "The demo has reached today's usage budget. "
                                     "Please try again tomorrow.")

        t0 = time.perf_counter()
        p = app.state.pipeline
        try:
            res = p.answer(body.question)
        except Exception as e:  # upstream API outage, timeout, etc.
            logger.log({**base, "status": 502, "error": f"{type(e).__name__}: {e}",
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 1)})
            raise HTTPException(502, "The answer service is temporarily unavailable.") from e

        budget.add(res.cost_usd)
        latency = round((time.perf_counter() - t0) * 1000, 1)
        logger.log({
            **base, "status": 200, "coverage": res.coverage, "model": res.model,
            "retrieved": [{"chunk_id": r["chunk_id"], "score": r["score"]} for r in res.retrieved],
            "citations": [{"source": c.source, "chunk_id": c.chunk_id,
                           "quote_verified": c.quote_verified} for c in res.citations],
            "timings_ms": {**res.timings_ms, "endpoint_total": latency},
            "usage": asdict(res.usage), "cost_usd": res.cost_usd,
            "stop_reason": res.stop_reason,
            "error": res.error, "llm_request_id": res.request_id,
        })
        retrieved_at = getattr(p, "retrieved_at", {})
        return AskResponse(
            request_id=rid, answer=res.answer, coverage=res.coverage, latency_ms=latency,
            citations=[CitationOut(source=c.source, title=c.title, url=c.url, quote=c.quote,
                                   quote_verified=c.quote_verified,
                                   retrieved_at=retrieved_at.get(
                                       c.chunk_id.split("#")[0], "")[:10])
                       for c in res.citations if c.chunk_id],  # drop citations to unknown ids
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app


def __getattr__(name: str):
    # `uvicorn visaguide.api.main:app` builds the real app on first access, so that
    # importing this module (e.g. in tests) doesn't load models or need an API key.
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)

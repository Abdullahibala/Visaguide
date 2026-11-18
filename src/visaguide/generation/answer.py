"""The full RAG pipeline: question -> retrieve -> prompt -> LLM -> checked, cited answer."""

import csv
import time
from dataclasses import asdict, dataclass, field

from visaguide.chunking import Chunk
from visaguide.config import Config
from visaguide.evaluation.anchor import normalise
from visaguide.generation.llm import LLMClient, Usage, make_llm
from visaguide.generation.prompts import (
    ANSWER_SCHEMA,
    SYSTEM_PROMPT,
    build_user_prompt,
    source_id,
)
from visaguide.retrieval.retriever import Retriever


@dataclass
class Citation:
    source: str               # "S2"
    chunk_id: str             # "" if the model cited an id that doesn't exist
    url: str
    title: str
    quote: str
    quote_verified: bool      # quote really appears in the cited chunk


@dataclass
class AnswerResult:
    question: str
    answer: str
    refused: bool
    citations: list[Citation]
    retrieved: list[dict]     # chunk_id, doc_id, start, end, title, url, score
    model: str
    stop_reason: str
    usage: Usage
    cost_usd: float | None
    timings_ms: dict = field(default_factory=dict)
    error: str = ""
    request_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def verify_citations(raw: list[dict], chunks: list[Chunk]) -> list[Citation]:
    by_id = {source_id(i): c for i, c in enumerate(chunks, 1)}
    out = []
    for cit in raw:
        sid = str(cit.get("source", "")).strip().strip("[]")
        quote = str(cit.get("quote", ""))
        chunk = by_id.get(sid)
        verified = bool(chunk and normalise(quote) and normalise(quote) in normalise(chunk.text))
        out.append(Citation(source=sid, chunk_id=chunk.chunk_id if chunk else "",
                            url=chunk.url if chunk else "", title=chunk.title if chunk else "",
                            quote=quote, quote_verified=verified))
    return out


class RAGPipeline:
    def __init__(self, cfg: Config, retriever: Retriever, llm: LLMClient | None = None,
                 mode: str | None = None, top_k: int | None = None):
        self.cfg = cfg
        self.retriever = retriever
        self.llm = llm or make_llm(cfg.generation)
        self.mode = mode or cfg.retrieval.default_mode
        self.top_k = top_k or cfg.retrieval.top_k
        with cfg.paths.manifest.open(encoding="utf-8") as f:
            self.retrieved_at = {r["doc_id"]: r["retrieved_at"] for r in csv.DictReader(f)}

    @classmethod
    def from_config(cls, cfg: Config, chunking: str | None = None, **kw) -> "RAGPipeline":
        chunking = chunking or cfg.retrieval.default_chunking
        return cls(cfg, Retriever.load(cfg.paths.index_dir / chunking, cfg.retrieval), **kw)

    def answer(self, question: str) -> AnswerResult:
        t0 = time.perf_counter()
        hits = self.retriever.search(question, mode=self.mode, k=self.top_k)
        t_retrieval = (time.perf_counter() - t0) * 1000
        chunks = [h.chunk for h in hits]

        user = build_user_prompt(question, chunks, self.retrieved_at)
        resp = self.llm.generate_json(SYSTEM_PROMPT, user, ANSWER_SCHEMA)
        price = self.cfg.pricing_usd_per_mtok.get(resp.model)

        data = resp.data or {}
        if resp.data is None:
            # Model-level refusal or broken output: fail closed, never show a guess.
            answer = "Sorry, I couldn't produce a reliable answer to this question."
            refused, citations = True, []
        else:
            answer = str(data.get("answer", "")).strip()
            refused = bool(data.get("refused", False))
            citations = verify_citations(data.get("citations") or [], chunks)

        return AnswerResult(
            question=question, answer=answer, refused=refused, citations=citations,
            retrieved=[{"chunk_id": h.chunk.chunk_id, "doc_id": h.chunk.doc_id,
                        "start": h.chunk.start, "end": h.chunk.end, "title": h.chunk.title,
                        "url": h.chunk.url, "score": round(h.score, 4)} for h in hits],
            model=resp.model, stop_reason=resp.stop_reason, usage=resp.usage,
            cost_usd=resp.usage.cost_usd(price),
            timings_ms={"retrieval": round(t_retrieval, 1),
                        "generation": round(resp.latency_ms, 1),
                        "total": round((time.perf_counter() - t0) * 1000, 1),
                        **{f"retrieval_{k}": v for k, v in
                           self.retriever.last_timings_ms.items()}},
            error=resp.error, request_id=resp.request_id,
        )

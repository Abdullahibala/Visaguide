"""LLM access behind one small interface, so the provider and model are a config change.

Everything else in the project (answer pipeline, judge) depends only on
`LLMClient.generate_json`. Adding another provider means writing one class with
that method and one line in make_llm(). No framework needed.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Protocol

from visaguide.config import LLMCfg, Price


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0          # includes thinking tokens: they are billed as output
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def cost_usd(self, price: Price | None) -> float | None:
        if price is None:
            return None
        return (self.input_tokens * price.input + self.output_tokens * price.output
                + self.cache_read_tokens * price.cache_read
                + self.cache_write_tokens * price.cache_write) / 1_000_000


@dataclass
class LLMResponse:
    data: dict | None               # parsed JSON; None if the model refused or output was cut
    model: str                      # model that actually served the request
    stop_reason: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    request_id: str = ""
    error: str = ""


class LLMClient(Protocol):
    def generate_json(self, system: str, user: str, schema: dict) -> LLMResponse: ...


class AnthropicClient:
    """Claude via the official SDK, with structured JSON output."""

    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, cfg: LLMCfg):
        import anthropic  # imported here so tests and other providers don't need it

        self.cfg = cfg
        # No key argument: the SDK reads ANTHROPIC_API_KEY (from .env via config.py)
        # or an `ant auth login` profile. Keys never appear in code.
        self.client = anthropic.Anthropic(max_retries=3)

    def generate_json(self, system: str, user: str, schema: dict) -> LLMResponse:
        kwargs = dict(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": self.cfg.effort,
                # Constrained decoding: the reply is guaranteed to be JSON matching
                # the schema, so no fragile regex parsing of free text.
                "format": {"type": "json_schema", "schema": schema},
            },
        )
        t0 = time.perf_counter()
        if self.cfg.fallbacks:
            # If the model declines on safety grounds, the API reruns the request on
            # a fallback model inside the same call ("default" picks it per category).
            resp = self.client.beta.messages.create(
                **kwargs, betas=[self.FALLBACK_BETA], fallbacks="default")
        else:
            resp = self.client.messages.create(**kwargs)
        latency = (time.perf_counter() - t0) * 1000

        u = resp.usage
        usage = Usage(
            input_tokens=u.input_tokens, output_tokens=u.output_tokens,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        out = LLMResponse(data=None, model=resp.model, stop_reason=resp.stop_reason or "",
                          usage=usage, latency_ms=latency,
                          request_id=getattr(resp, "_request_id", "") or "")
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            out.error = f"model refusal ({getattr(details, 'category', None)})"
            return out
        if resp.stop_reason == "max_tokens":
            out.error = "output truncated at max_tokens"
            return out
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            out.data = json.loads(text)
        except json.JSONDecodeError as e:
            out.error = f"invalid JSON: {e}"
        return out


def make_llm(cfg: LLMCfg) -> LLMClient:
    if cfg.provider == "anthropic":
        return AnthropicClient(cfg)
    raise ValueError(f"Unknown LLM provider '{cfg.provider}'. Add a client class in llm.py.")

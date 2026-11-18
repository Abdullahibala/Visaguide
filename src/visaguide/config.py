"""Configuration: non-secret settings from YAML, secrets from environment / .env.

Why split them? The YAML is committed so every experiment is reproducible from git;
secrets must never be committed, so they only come from the environment.
"""

from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
# Put .env values into os.environ (where SDKs look for keys). override=False: real
# environment variables win, which is how secrets arrive in Docker / Cloud Run.
load_dotenv(ROOT / ".env", override=False)


class Paths(BaseModel):
    sources: Path
    raw_dir: Path
    processed: Path
    manifest: Path
    index_dir: Path

    def resolve(self) -> "Paths":
        return Paths(**{k: ROOT / v for k, v in self.model_dump().items()})


class IngestCfg(BaseModel):
    user_agent: str
    timeout_s: float = 30
    delay_per_domain_s: float = 2.0
    min_chars: int = 400


class FixedChunkCfg(BaseModel):
    size_words: int = 200
    overlap_words: int = 40


class StructureChunkCfg(BaseModel):
    max_words: int = 200
    min_words: int = 40


class ChunkingCfg(BaseModel):
    fixed: FixedChunkCfg = FixedChunkCfg()
    structure: StructureChunkCfg = StructureChunkCfg()


class RetrievalCfg(BaseModel):
    embedding_model: str
    query_prefix: str = ""
    reranker_model: str
    candidates: int = 50
    rrf_k: int = 60
    rrf_weights: list[float] = [1.0, 1.0]  # [dense, bm25]
    rerank_top_n: int = 30
    default_chunking: str = "structure"
    default_mode: str = "dense_rerank"
    top_k: int = 5


class LLMCfg(BaseModel):
    provider: str = "anthropic"
    model: str
    effort: str = "medium"
    max_tokens: int = 16000
    fallbacks: bool = True


class Price(BaseModel):
    input: float
    output: float
    cache_read: float = 0.0
    cache_write: float = 0.0


class ApiCfg(BaseModel):
    max_question_chars: int = 500
    rate_limit_per_minute: int = 5
    rate_limit_per_day: int = 40
    daily_budget_usd: float = 2.0
    log_path: Path = Path("logs/queries.jsonl")
    trust_forwarded_for: bool = True


class Config(BaseModel):
    paths: Paths
    ingest: IngestCfg
    chunking: ChunkingCfg
    retrieval: RetrievalCfg
    generation: LLMCfg
    judge: LLMCfg
    api: ApiCfg = ApiCfg()
    pricing_usd_per_mtok: dict[str, Price] = {}


class Secrets(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")
    anthropic_api_key: str = ""
    # Salt for hashing client IPs in logs. Without a secret salt, the hash of an IPv4
    # address can be reversed by brute force (there are only 2^32 addresses).
    ip_hash_salt: str = "dev-only-salt-change-me"


@lru_cache
def load_config(path: str | Path = ROOT / "configs" / "default.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = Config(**raw)
    cfg.paths = cfg.paths.resolve()
    return cfg

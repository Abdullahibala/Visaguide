"""Configuration: non-secret settings from YAML, secrets from environment / .env.

Why split them? The YAML is committed so every experiment is reproducible from git;
secrets must never be committed, so they only come from the environment.
"""

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Paths(BaseModel):
    sources: Path
    raw_dir: Path
    processed: Path
    manifest: Path

    def resolve(self) -> "Paths":
        return Paths(**{k: ROOT / v for k, v in self.model_dump().items()})


class IngestCfg(BaseModel):
    user_agent: str
    timeout_s: float = 30
    delay_per_domain_s: float = 2.0
    min_chars: int = 400


class Config(BaseModel):
    paths: Paths
    ingest: IngestCfg


class Secrets(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")
    anthropic_api_key: str = ""


@lru_cache
def load_config(path: str | Path = ROOT / "configs" / "default.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = Config(**raw)
    cfg.paths = cfg.paths.resolve()
    return cfg

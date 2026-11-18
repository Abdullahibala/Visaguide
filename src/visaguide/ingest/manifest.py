"""The manifest: one row per source, recording where it came from and when.

It answers "where does this answer come from, and how old is it?". The UI shows
retrieved_at next to each citation because visa rules change.
"""

import csv
from pathlib import Path

FIELDS = [
    "doc_id", "country", "source_type", "topic", "title", "url", "final_url",
    "format", "retrieved_at", "sha256", "n_chars", "status", "note",
]


def write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["country"], r["source_type"], r["doc_id"]))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {r["url"]: r for r in csv.DictReader(f)}

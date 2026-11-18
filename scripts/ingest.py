"""M1 pipeline: sources.yaml -> raw snapshots -> clean text -> docs.jsonl + manifest.csv.

Usage:
    python scripts/ingest.py             # fetch only sources not fetched yet, then clean all
    python scripts/ingest.py --refresh   # re-fetch everything (new retrieved_at dates)

Fetched pages are cached in data/raw/, so re-running the cleaner never hits the sites again.
"""

import argparse
import json
from collections import Counter, defaultdict
from urllib.parse import urlparse

import yaml

from visaguide.config import load_config
from visaguide.ingest.clean import extract, remove_repeated_blocks
from visaguide.ingest.fetch import PoliteFetcher, doc_id_for
from visaguide.ingest.manifest import read_manifest, write_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-fetch all sources")
    args = ap.parse_args()

    cfg = load_config()
    sources = yaml.safe_load(cfg.paths.sources.read_text(encoding="utf-8"))
    previous = read_manifest(cfg.paths.manifest)
    fetcher = PoliteFetcher(cfg.ingest.user_agent, cfg.ingest.timeout_s,
                            cfg.ingest.delay_per_domain_s)

    rows, docs, failures, short = [], [], [], []
    for i, src in enumerate(sources, 1):
        url, country = src["url"], src["country"]
        doc_id = doc_id_for(url, country)
        cached = [p for p in cfg.paths.raw_dir.glob(f"{doc_id}.*")]
        prev = previous.get(url)

        if cached and prev and prev.get("status") == "200" and not args.refresh:
            raw_path = cached[0]
            meta = {k: prev[k] for k in ("final_url", "retrieved_at", "sha256")}
        else:
            print(f"[{i}/{len(sources)}] fetching {url}")
            res = fetcher.fetch(url, cfg.paths.raw_dir, doc_id)
            if res.error:
                failures.append((url, res.error))
                continue
            raw_path = res.raw_path
            meta = {"final_url": res.final_url, "retrieved_at": res.retrieved_at,
                    "sha256": res.sha256}

        title, text = extract(raw_path, url)
        title = title or src.get("title", "")

        row = {
            "doc_id": doc_id, "country": country, "source_type": src["source_type"],
            "topic": src.get("topic", ""), "title": title, "url": url,
            "format": "pdf" if raw_path.suffix == ".pdf" else "html",
            "n_chars": len(text), "status": 200, "note": src.get("note", ""), **meta,
        }
        rows.append(row)
        docs.append({**row, "text": text})

    # Cross-page boilerplate removal, one site (domain) at a time.
    by_domain = defaultdict(list)
    for d in docs:
        by_domain[urlparse(d["url"]).netloc].append(d)
    for group in by_domain.values():
        for d, text in zip(group, remove_repeated_blocks([d["text"] for d in group]),
                           strict=True):
            d["text"], d["n_chars"] = text, len(text)
    for row, d in zip(rows, docs, strict=True):
        row["n_chars"] = d["n_chars"]
        if d["n_chars"] < cfg.ingest.min_chars:
            short.append((d["url"], d["n_chars"]))

    cfg.paths.processed.parent.mkdir(parents=True, exist_ok=True)
    with cfg.paths.processed.open("w", encoding="utf-8") as f:
        for d in sorted(docs, key=lambda d: d["doc_id"]):
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    write_manifest(rows, cfg.paths.manifest)

    # ---- summary ----
    print(f"\nOK: {len(docs)} documents, {sum(d['n_chars'] for d in docs):,} chars total")
    counts = Counter((d["country"], d["source_type"]) for d in docs)
    for (country, stype), n in sorted(counts.items()):
        print(f"  {country:8s} {stype:11s} {n}")
    if short:
        print(f"\nFLAGGED short (<{cfg.ingest.min_chars} chars) - inspect or drop:")
        for url, n in short:
            print(f"  {n:6d}  {url}")
    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for url, err in failures:
            print(f"  {err:30s} {url}")


if __name__ == "__main__":
    main()

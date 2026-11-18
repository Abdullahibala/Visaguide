"""Turn the hand-written eval/testset_draft.yaml into eval/testset.jsonl.

For every gold quote it finds exact character offsets in the processed corpus. It
refuses to write anything if a quote can't be found, so a typo can't silently
become a question that can never be answered.

    python scripts/anchor_testset.py
"""

import hashlib
import json
import sys
from collections import Counter

import yaml

from visaguide.config import ROOT, load_config
from visaguide.evaluation.anchor import find_quote, normalise

DRAFT = ROOT / "eval" / "testset_draft.yaml"
OUT = ROOT / "eval" / "testset.jsonl"
DEV_PERCENT = 30
COUNTRIES = {"germany", "turkiye", "both"}
TYPES = {"factoid", "procedural", "list", "conditional", "comparison"}


def split_for(qid: str) -> str:
    """Stable dev/test assignment: a question never changes split when others are added."""
    bucket = int(hashlib.sha1(qid.encode()).hexdigest(), 16) % 100
    return "dev" if bucket < DEV_PERCENT else "test"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    with cfg.paths.processed.open(encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    by_id = {d["doc_id"]: d for d in docs}
    by_url = {d["url"]: d for d in docs}

    items = yaml.safe_load(DRAFT.read_text(encoding="utf-8")) or []
    errors, warnings, out = [], [], []
    seen_ids = set()

    for it in items:
        qid = it.get("id", "?")
        where = f"[{qid}]"
        if qid in seen_ids:
            errors.append(f"{where} duplicate id")
        seen_ids.add(qid)
        for field in ("question", "answerable", "country", "type", "answer"):
            if field not in it:
                errors.append(f"{where} missing field '{field}'")
        if it.get("country") not in COUNTRIES:
            errors.append(f"{where} country must be one of {sorted(COUNTRIES)}")
        if it.get("type") not in TYPES:
            errors.append(f"{where} type must be one of {sorted(TYPES)}")
        gold_in = it.get("gold") or []
        if it.get("answerable") and not gold_in:
            errors.append(f"{where} answerable question needs at least one gold passage")
        if it.get("answerable") is False and gold_in:
            errors.append(f"{where} unanswerable question must not have gold passages")

        gold = []
        for j, g in enumerate(gold_in):
            ref, quote = str(g.get("doc", "")), g.get("quote", "")
            doc = by_id.get(ref) or by_url.get(ref)
            if doc is None:
                errors.append(f"{where} gold #{j + 1}: unknown doc '{ref}' (use doc_id or URL)")
                continue
            if len(normalise(quote)) < 20:
                errors.append(f"{where} gold #{j + 1}: quote too short to be unambiguous")
                continue
            found = find_quote(doc["text"], quote)
            if found is None:
                elsewhere = [d["doc_id"] for d in docs if find_quote(d["text"], quote)]
                hint = f" - but found in {elsewhere}" if elsewhere else ""
                errors.append(f"{where} gold #{j + 1}: quote not found in {doc['doc_id']}{hint}"
                              f"\n      quote: {quote[:120]!r}")
                continue
            start, end, n = found
            if n > 1:
                warnings.append(f"{where} gold #{j + 1}: quote occurs {n}x in {doc['doc_id']};"
                                " using the first - lengthen it if that's the wrong one")
            gold.append({"doc_id": doc["doc_id"], "start": start, "end": end,
                         "group": str(g.get("group", f"g{j + 1}")),
                         "quote": doc["text"][start:end]})

        out.append({
            "id": qid, "question": it.get("question", "").strip(),
            "answerable": bool(it.get("answerable")), "country": it.get("country"),
            "type": it.get("type"), "reference_answer": str(it.get("answer", "")).strip(),
            "gold": gold, "split": it.get("split") or split_for(qid),
            "notes": it.get("notes", ""),
        })

    for w in warnings:
        print("WARN ", w)
    if errors:
        print(f"\n{len(errors)} error(s) - nothing written:")
        for e in errors:
            print("ERROR", e)
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for q in out:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    c = Counter((q["split"], "answerable" if q["answerable"] else "unanswerable") for q in out)
    print(f"Wrote {len(out)} questions to {OUT.relative_to(ROOT)}")
    for (split, kind), n in sorted(c.items()):
        print(f"  {split:5s} {kind:13s} {n}")
    print("  by country:", dict(Counter(q["country"] for q in out)))
    print("  by type:   ", dict(Counter(q["type"] for q in out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

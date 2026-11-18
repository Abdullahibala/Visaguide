"""Suggest other documents that may state the same fact as each gold quote.

For every gold quote in eval/testset.jsonl, BM25-search the structure chunks of the
OTHER documents and print the top candidates. A human then decides which ones really
are alternatives and adds them to the same group in testset_draft.yaml.
Missing alternatives make recall look worse than it is.

    python scripts/find_alternatives.py            # all answerable questions
    python scripts/find_alternatives.py q001 q012  # selected questions
"""

import json
import sys

from visaguide.config import ROOT, load_config
from visaguide.retrieval.bm25 import BM25Retriever


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    with (cfg.paths.index_dir / "structure" / "chunks.jsonl").open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    bm25 = BM25Retriever([c["text"] for c in chunks])
    with (ROOT / "eval" / "testset.jsonl").open(encoding="utf-8") as f:
        qs = [q for q in map(json.loads, f) if q["answerable"]]
    if len(sys.argv) > 1:
        qs = [q for q in qs if q["id"] in sys.argv[1:]]

    for q in qs:
        gold_docs = {g["doc_id"] for g in q["gold"]}
        print(f"\n=== {q['id']}  {q['question']}")
        for g in q["gold"]:
            print(f"  gold[{g['group']}] {g['doc_id']}: {' '.join(g['quote'].split())[:110]}")
            for i, score in bm25.search(g["quote"], k=12):
                c = chunks[i]
                if c["doc_id"] in gold_docs:
                    continue
                print(f"    ? {score:5.1f} {c['doc_id']} {c['title'][:40]:40s} | "
                      f"{' '.join(c['text'].split())[:150]}")
                gold_docs.add(c["doc_id"])  # one suggestion per document
                if len(gold_docs) > len({g['doc_id'] for g in q['gold']}) + 3:
                    break


if __name__ == "__main__":
    main()

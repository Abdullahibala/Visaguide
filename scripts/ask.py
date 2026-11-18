"""Ask VisaGuide one question from the command line (calls the LLM API: costs money).

    python scripts/ask.py "How much money do I need in my blocked account?"
"""

import argparse
import sys

from visaguide.config import load_config
from visaguide.generation.answer import RAGPipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--chunking")
    ap.add_argument("--mode")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = load_config()
    res = RAGPipeline.from_config(cfg, chunking=args.chunking, mode=args.mode).answer(args.question)

    print(f"\n{'[REFUSED] ' if res.refused else ''}{res.answer}\n")
    for c in res.citations:
        mark = "verified" if c.quote_verified else "NOT FOUND IN SOURCE"
        print(f"  [{c.source}] {c.title}\n       {c.url}\n       \"{c.quote}\" ({mark})")
    cost = f"${res.cost_usd:.4f}" if res.cost_usd is not None else "unknown (model not priced)"
    print(f"\nmodel={res.model}  tokens in/out={res.usage.input_tokens}/"
          f"{res.usage.output_tokens}  cost={cost}  timings(ms)={res.timings_ms}")
    if res.error:
        print(f"ERROR: {res.error}")
    print("\nNot legal advice. Rules change: confirm with the responsible authority.")


if __name__ == "__main__":
    main()

# Test set annotation guidelines

The test set is the yardstick for every number in the README, so it's written by
hand and frozen before any tuning.

## Target composition (~80 questions)

| | count |
|---|---|
| answerable | ~65 (about half Germany, half Türkiye, a few `both`) |
| unanswerable | ~15 |

Mix the `type`s: **factoid** (one fact: an amount, a number of days), **procedural**
(steps/order), **list** (required documents), **conditional** ("does it depend on…"),
**comparison** (Germany vs Türkiye, needs passages from both).

## Rules

1. **Write the question before you look for the passage, in your own words.** Copying
   the source's wording makes BM25 look much better than it would on real questions
   (lexical leakage). Ask it the way a confused student would.
2. **Quote the smallest passage that fully answers it**, usually 1-3 sentences.
   Whole sections make every chunk "relevant" and inflate recall.
3. **Use groups honestly.** If several sources state the same fact, add them under the
   same group. Missing alternatives make the system look worse than it is.
4. **Spread questions across sources.** Don't take 10 questions from one FAQ page.
   Cover all topics: student visa, blocked account / finances, health insurance,
   residence permit, address registration, enrolment, work rules, post-study, fees.
5. **Include the hard cases on purpose:**
   - Conflicting or outdated sources (e.g. the blocked-account amount differs between
     pages from different years). Put the current figure in the reference answer and
     explain in `notes`.
   - Questions needing two facts from different places (two groups).
   - Questions where the answer is in a PDF or the long Residence Act translation.
6. **Unanswerable questions should be plausible near-misses**: on-topic, but the
   specific detail isn't in the corpus (a specific office's processing time, a fee
   for a year not covered, a country-specific rule we didn't collect). Verify by
   searching the corpus before marking `answerable: false`. A question nobody would
   ask ("what's the capital of France") tests nothing.
7. **Never edit questions after seeing results on the `test` split.** Fix problems in
   the `dev` split. If a test question is truly broken (wrong gold), fix it and note
   it in `notes`.

## Provenance and revision history

- **Draft (80 questions):** written by the author.
- **Audit + revision (2026-11-18), LLM-assisted:** an automated audit found 7 questions
  labelled unanswerable that the corpus does answer, 10 near-duplicate pairs (7 crossing
  the dev/test split, which leaks dev into test), gold passages that didn't support
  their reference answers, and questions naming their source. Fixes: 15 questions
  removed (listed at the top of `testset_draft.yaml`), 2 relabelled as answerable,
  16 added from previously uncovered documents, alternatives added after checking
  `scripts/find_alternatives.py` suggestions. Every quote was taken verbatim from the
  corpus and checked by `anchor_testset.py`. All of this happened **before any
  evaluation run on the test split**.
- **Result:** 81 questions (64 answerable, 17 unanswerable), 41 distinct gold documents.

This is reported as a limitation in the main README: part of the test set was
revised with LLM help, so it isn't a purely human-written benchmark.

## Workflow

1. Find passages: `.venv\Scripts\python scripts\search.py "..."` or open
   `data/processed/docs.jsonl`. `doc_id`s are in `data/manifest.csv`.
2. Add the item to `eval/testset_draft.yaml`.
3. Run `.venv\Scripts\python scripts\anchor_testset.py`. It refuses to write if any
   quote can't be found, and tells you where else the quote occurs.

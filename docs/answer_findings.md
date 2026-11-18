# Answer-quality findings (M4)

Final configuration (chosen on the dev split): structure chunking, `dense_rerank`,
top_k 8, generator `claude-opus-5-5` (effort medium), judge `claude-sonnet-5-5`.

Test run: `eval/results/answers_test_structure_dense_rerank_final_20261119-012330`
(50 questions: 41 answerable, 9 unanswerable). Numbers are reported **as labelled**;
known label problems are listed below, not silently fixed.

## Results

| Metric | Dev v1 (k=5) | Dev v2 (k=5) | Dev v3 (k=8) | **Test (k=8)** |
|---|---|---|---|---|
| Correctness (answerable; correct=1, partial=0.5) | 0.739 | 0.750 | 0.812 | **0.951** |
| Share fully correct | 0.739 | — | — | **0.902** |
| Faithfulness (claims supported by sources) | 0.967 | 0.980 | 0.966 | **0.984** |
| Citation accuracy (claims supported by cited source) | 0.934 | 0.966 | 0.950 | **0.957** |
| Quotes verified verbatim in cited chunk | 0.992 | 0.983 | 0.993 | **0.989** |
| Unanswerable: gap flagged (coverage none/partial) | — | 1.000 | 1.000 | **0.778** (7/9) |
| Unanswerable: refused outright | 0.750 | 0.857 | 0.857 | **0.333** (3/9) |
| Answerable: false refusals | 0.087 | 0.042 | 0.042 | **0.000** |
| Latency total p50 / p95 | 6.8 / 9.6 s | 6.8 / 9.6 s | 6.8 / 9.5 s | **7.4 / 10.3 s** |
| Generation latency p50 / p95 | | | | 5.9 / 9.1 s |
| Retrieval latency p50 (CPU, incl. reranker) | | | | 1.4 s |
| Cost per query, mean / p95 | $0.021 | $0.021 | $0.026 | **$0.028 / $0.034** |

Dev v1 had 23 answerable and 8 unanswerable questions; v2 and v3 have 24 and 7
(q049 relabelled). Cost is generation only, computed from API token usage x the
config prices. Judge cost on the test run was $0.79 in total. The test set scores
higher than dev; with n = 41 and n = 24, part of that is which questions landed in
which split (dev holds several hard cases: a German-language passage, comparison
questions).

## What changed on dev, and why

- **v1 → v2:** a yes/no `refused` flag scored the ideal near-miss answer ("no figure
  for Izmir; the general range is 30–90 days") as a failure. Replaced with
  `coverage` = full / partial / none. The judge stopped counting "the sources don't
  say X" as an unsupported claim. q049 relabelled as answerable.
- **v2 → v3 (top_k 5 → 8):** comparison and multi-part questions lost one half at
  k = 5. With 8 chunks, 3 questions improved and none regressed (paired bootstrap
  +0.062 [+0.000, +0.125]), for +$0.005 per query.

## Error analysis: the 10 worst test cases

`python scripts/error_analysis.py --run <test run>` ranks failures by severity
(answering an unanswerable question > confidently wrong > unsupported claim >
false refusal > partial). Each case below was checked against the retrieved text.

| Failure type | Cases | What happened |
|---|---|---|
| **Test-label error**: labelled unanswerable, but the corpus answers it | q066, q076 | q076: Service Berlin states "An extension... for more than 18 months for the purpose of further job search is not possible." q066: the Miami consulate lists different fees by nationality. The model answered correctly with citations; the metric counts both as "answered an unanswerable question". |
| **Source conflict** | q025, q061 | q025: METU says the post-graduation permit "can be extended for an additional one year"; the migration authority says it is issued "only once". q061: Munich says permits are usually granted for 3 years, Hamburg says at most 2. The model reported the versions with citations; the judge scored against the single reference. |
| **Judge error** | q065 | The judge marked "students pay low contributions only if they earn more than 556 EUR per month" as unsupported, but it is in the retrieved DAAD chunk. |
| **Minor unsupported inference** in an otherwise correct answer | q039, q086, q092 | e.g. "isoidb.ankara.edu.tr appears to be the international students office". Small hedged inferences, not wrong facts. The real generation weakness found. |
| **Incomplete retrieval → partial answer** | q036, q095 | The gold page wasn't retrieved. The answer was built from other sources and was roughly right but vaguer (">90 days" instead of "3 months or longer"). q095 is one of the incomplete-gold cases from M3. |

**Main finding:** no answer was confidently wrong about a fact the sources state
clearly. The most harmful failure type, answering an unanswerable question, came
from two **label errors**, not hallucination. With those two relabelled, 7 of 7
unanswerable questions would have the gap flagged. The scores above are kept as
labelled, because correcting only the cases where the system "failed" would
bias the evaluation upward.

**What this means for the evaluation itself:**
1. The test set has label errors (q066, q076), even after the audit. A second,
   independent annotation pass is needed for publication-grade numbers.
2. Single reference answers can't represent source conflicts. Conflict questions
   need references listing every version (planned for the test set's next revision).
3. The judge makes mistakes (q065). Its agreement with human labels is measured in
   `judge_agreement.md` before any judge-based number is trusted.

## Limitations

- n = 41 answerable / 9 unanswerable test questions: confidence intervals are wide.
- Retrieval latency is laptop CPU; the reranker dominates it.
- Generation latency (~6 s p50) is the main user-facing cost. Streaming the answer
  (M5) hides most of it; a smaller model would cut it at some quality cost.
- Part of the test set was revised with LLM assistance (see `eval/README.md`).

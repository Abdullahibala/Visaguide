# Retrieval findings (M3)

Final run: `eval/results/retrieval_test_final_20261119-001749` (test split, n = 41
answerable questions; paired comparisons in `paired_comparisons.md` of that folder).
Dev-split runs (`retrieval_dev_*`, n = 23) were used for every decision below.

## Results (test split)

| Chunking | Retriever | R@1 | R@5 [95% CI] | R@10 | MRR [95% CI] | p50 ms |
|---|---|---|---|---|---|---|
| fixed | dense | 0.439 | 0.817 [0.70-0.93] | 0.866 | 0.579 [0.45-0.69] | 18 |
| fixed | bm25 | 0.280 | 0.524 [0.38-0.67] | 0.634 | 0.393 [0.27-0.52] | 6 |
| fixed | hybrid | 0.305 | 0.659 [0.52-0.79] | 0.841 | 0.489 [0.37-0.61] | 24 |
| fixed | hybrid_rerank | 0.451 | 0.780 [0.66-0.90] | 0.829 | 0.610 [0.49-0.73] | 977 |
| fixed | dense_rerank | 0.451 | 0.780 [0.66-0.90] | 0.829 | 0.611 [0.49-0.73] | 1016 |
| structure | dense | 0.366 | 0.756 [0.61-0.88] | 0.854 | 0.558 [0.44-0.67] | 17 |
| structure | bm25 | 0.220 | 0.488 [0.34-0.65] | 0.573 | 0.322 [0.21-0.44] | 4 |
| structure | hybrid | 0.244 | 0.707 [0.57-0.84] | 0.829 | 0.438 [0.34-0.55] | 24 |
| structure | hybrid_rerank | 0.476 | 0.793 [0.66-0.91] | 0.854 | 0.638 [0.51-0.77] | 936 |
| structure | dense_rerank | 0.500 | 0.768 [0.63-0.89] | 0.854 | 0.644 [0.52-0.77] | 943 |

Latency: CPU only (laptop), excludes the LLM call.

## Findings (significance from the paired bootstrap, 10,000 resamples)

1. **Dense retrieval beats BM25 by a wide margin** (fixed: R@5 +0.29, significant).
   Questions are written in students' own words (mean question/quote word overlap 0.31),
   and BM25 needs lexical overlap; it also has no stemming.
2. **Equal-weight RRF hybrid did not help.** On test it was significantly *worse* than
   dense for one metric per chunking (fixed R@5 −0.16; structure MRR −0.12). On dev it
   tied with dense. Down-weighting BM25 (0.5, 0.25) changed nothing on dev, so the
   weights stay at 1/1. Lesson: fusion helps when retrievers are comparably strong;
   a much weaker one adds noise.
3. **The cross-encoder fixes the hybrid's ordering**: MRR +0.12 (fixed) and +0.20
   (structure) over hybrid, both significant.
4. **After reranking, BM25 adds nothing**: dense_rerank vs hybrid_rerank, ΔMRR +0.006
   [−0.030, +0.042]. Chosen default: **structure + dense_rerank** (decided on dev:
   equal quality, one fewer component; heading paths make citations readable).
5. **Reranking vs plain dense is not significant at n = 41** (structure: ΔMRR +0.086
   [−0.040, +0.215]) and costs ~50x latency on CPU. Honest reading: the reranker
   *probably* helps ranking (consistent direction on dev and test), but this test set
   is too small to prove it.
6. **Chunking strategy made no measurable difference** (all paired CIs include 0).

## Error analysis: questions never retrieved in the top 10

`python scripts/diagnose_question.py q016 q025 q061 q095`

| Q | Finding | Type |
|---|---|---|
| q095 | Heidelberg's "...the United States do not require a visa..." ranked #2 but is not labelled gold | gold incomplete |
| q061 | Hamburg's "maximum of two years and must therefore be regularly extended" ranked #1 | gold incomplete |
| q025 | METU's "After you graduate: Turkish laws allow students who graduate..." ranked #3 | gold incomplete |
| q016 | Answer is one sentence inside a long step-by-step page. Dense rank 153 (fixed) vs 37 (structure) | real retrieval failure |

**Recall is therefore underestimated.** The alternatives finder used BM25, so it missed
paraphrased alternatives. These three are deliberately *not* patched by hand: fixing
only failing questions would bias the metrics upward. The planned fix is TREC-style
**pooled relevance judgement**: pool the top-k of every configuration for every
question, judge all pooled chunks (LLM judge, validated against human labels), and
recompute.

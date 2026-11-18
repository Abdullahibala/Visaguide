Retrieval results - dev split, n = 23 answerable questions. Brackets: 95% bootstrap CI.

| Chunking | Retriever | R@1 | R@3 | R@5 | R@10 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| fixed | hybrid | 0.326 | 0.522 | 0.565 [0.37-0.76] | 0.652 | 0.470 [0.30-0.64] | 23 | 28 |
| fixed | hybrid_rerank | 0.370 | 0.522 | 0.630 [0.46-0.80] | 0.674 | 0.508 [0.34-0.68] | 1029 | 1507 |
| structure | hybrid | 0.283 | 0.413 | 0.543 [0.35-0.74] | 0.652 | 0.407 [0.25-0.58] | 24 | 29 |
| structure | hybrid_rerank | 0.370 | 0.609 | 0.652 [0.46-0.83] | 0.652 | 0.516 [0.34-0.69] | 1007 | 1271 |

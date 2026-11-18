Retrieval results - dev split, n = 23 answerable questions. Brackets: 95% bootstrap CI.

| Chunking | Retriever | R@1 | R@3 | R@5 | R@10 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| fixed | hybrid | 0.283 | 0.522 | 0.565 [0.37-0.76] | 0.739 | 0.453 [0.30-0.62] | 23 | 26 |
| fixed | hybrid_rerank | 0.370 | 0.522 | 0.630 [0.46-0.80] | 0.674 | 0.508 [0.34-0.68] | 986 | 1413 |
| structure | hybrid | 0.283 | 0.413 | 0.565 [0.37-0.76] | 0.696 | 0.419 [0.26-0.58] | 22 | 26 |
| structure | hybrid_rerank | 0.370 | 0.609 | 0.652 [0.46-0.83] | 0.652 | 0.518 [0.34-0.69] | 978 | 1266 |

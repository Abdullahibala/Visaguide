Retrieval results - dev split, n = 23 answerable questions. Brackets: 95% bootstrap CI.

| Chunking | Retriever | R@1 | R@3 | R@5 | R@10 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| fixed | dense | 0.239 | 0.391 | 0.565 [0.37-0.76] | 0.717 | 0.399 [0.25-0.55] | 17 | 21 |
| fixed | dense_rerank | 0.370 | 0.522 | 0.630 [0.46-0.80] | 0.674 | 0.507 [0.34-0.68] | 999 | 1398 |
| structure | dense | 0.217 | 0.391 | 0.478 [0.28-0.67] | 0.696 | 0.389 [0.23-0.56] | 17 | 21 |
| structure | dense_rerank | 0.370 | 0.609 | 0.652 [0.46-0.83] | 0.652 | 0.518 [0.34-0.69] | 982 | 1252 |

## LLM judge vs human labels

Run: `answers_test_structure_dense_rerank_final_20261119-012330`. Faithful (judge) = every claim supported_by_any.

### Faithfulness (yes/no) (n = 20)

- Raw agreement: **0.45**
- Cohen's kappa: **-0.10**

| Human | Judge | Count |
|---|---|---|
| no | yes | 10 |
| yes | no | 1 |
| yes | yes | 9 |

### Correctness (correct/partial/incorrect) (n = 20)

- Raw agreement: **0.95**
- Cohen's kappa: **0.64**

| Human | Judge | Count |
|---|---|---|
| correct | correct | 18 |
| correct | partial | 1 |
| partial | partial | 1 |

## LLM judge vs human labels

Run: `answers_test_structure_dense_rerank_final-judgev3_20261119-020537`. Labels: `human_labels.yaml`, field `faithful_shared_def`. Faithful (judge) = every claim supported_by_any.

### Faithfulness (yes/no) (n = 20)

- Raw agreement: **0.80**
- Cohen's kappa: **0.41**

| Human | Judge | Count |
|---|---|---|
| no | no | 2 |
| yes | no | 4 |
| yes | yes | 14 |

Judge recall on human-unfaithful answers: **2/2**

### Correctness (correct/partial/incorrect) (n = 20)

- Raw agreement: **0.95**
- Cohen's kappa: **0.65**

| Human | Judge | Count |
|---|---|---|
| correct | correct | 18 |
| correct | incorrect | 1 |
| partial | partial | 1 |

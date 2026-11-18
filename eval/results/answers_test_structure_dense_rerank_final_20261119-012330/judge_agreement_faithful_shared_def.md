## LLM judge vs human labels

Run: `answers_test_structure_dense_rerank_final_20261119-012330`. Labels: `human_labels.yaml`, field `faithful_shared_def`. Faithful (judge) = every claim supported_by_any.

### Faithfulness (yes/no) (n = 20)

- Raw agreement: **0.85**
- Cohen's kappa: **-0.07**

| Human | Judge | Count |
|---|---|---|
| no | yes | 2 |
| yes | no | 1 |
| yes | yes | 17 |

Judge recall on human-unfaithful answers: **0/2**

### Correctness (correct/partial/incorrect) (n = 20)

- Raw agreement: **0.95**
- Cohen's kappa: **0.64**

| Human | Judge | Count |
|---|---|---|
| correct | correct | 18 |
| correct | partial | 1 |
| partial | partial | 1 |

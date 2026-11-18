"""Agreement between the LLM judge and human labels.

Raw agreement is easy to read, but it is inflated when one label dominates: if 90% of
answers are "correct", two raters who always say "correct" agree 90% of the time
while measuring nothing. Cohen's kappa corrects for the agreement expected by chance:

    kappa = (p_observed - p_expected) / (1 - p_expected)

where p_expected = sum over labels of P(rater A says it) * P(rater B says it).
Rough reading (Landis & Koch): <0.2 slight, 0.2-0.4 fair, 0.4-0.6 moderate,
0.6-0.8 substantial, >0.8 almost perfect. With 20 items, the kappa estimate itself
is noisy, which is why we report both numbers and the confusion counts.
"""

from collections import Counter


def raw_agreement(a: list, b: list) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def cohen_kappa(a: list, b: list) -> float:
    p_o = raw_agreement(a, b)
    n = len(a)
    ca, cb = Counter(a), Counter(b)
    p_e = sum((ca[label] / n) * (cb[label] / n) for label in set(a) | set(b))
    if p_e == 1.0:  # both raters used a single identical label: agreement is trivial
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def confusion(a: list, b: list) -> dict[tuple, int]:
    return dict(Counter(zip(a, b, strict=True)))

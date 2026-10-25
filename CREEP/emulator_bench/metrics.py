"""EC level accuracy with the same definition as the CLIPZyme and Horizyn wrappers."""

from __future__ import annotations

import pandas as pd

K_VALUES = (1, 3, 5, 10, 20, 30, 40, 50)
LEVELS = (4, 3, 2, 1)


def matched_levels(true_ec: str, ranked: list[str]) -> int:
    """Longest EC prefix (0-4 levels) that the true EC shares with any ranked EC."""
    true_parts = true_ec.split(".")
    best = 0
    for ec in ranked:
        n = 0
        for a, b in zip(ec.split("."), true_parts):
            if a != b:
                break
            n += 1
        best = max(best, n)
    return best


def ec_level_accuracy(predictions: pd.DataFrame, k_values=K_VALUES) -> dict[str, dict[str, float]]:
    """Level-L accuracy at k in percent: share of true ECs matched to level L by the top-k ranked ECs.

    A reaction with several true ECs counts each EC with equal weight inside that reaction.
    """
    metrics = {}
    for k in k_values:
        hits = {level: [] for level in LEVELS}
        for true_ecs, ranked in zip(predictions["true_ecs"], predictions["ranked_ecs"]):
            matched = [matched_levels(ec, list(ranked[:k])) for ec in true_ecs]
            for level in LEVELS:
                hits[level].append(sum(m >= level for m in matched) / len(matched))
        metrics[f"k={k}"] = {f"level_{level}": 100 * sum(h) / len(h) for level, h in hits.items()}
    return metrics

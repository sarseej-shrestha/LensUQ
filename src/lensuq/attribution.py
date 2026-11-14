"""Orthogonal main-effect decomposition for complete balanced factorial designs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import prod

import numpy as np
import pandas as pd

INTERACTION = "interaction_residual"


@dataclass(frozen=True)
class Attribution:
    """Per-sample sums of squares, fractions, and factor-level probability means."""

    contributions: pd.DataFrame
    sums_of_squares: pd.DataFrame
    total_sum_of_squares: pd.Series
    level_means: pd.DataFrame
    perfectly_stable: pd.Series


def factorial_attribution(
    probabilities: pd.DataFrame, design: pd.DataFrame
) -> Attribution:
    """Decompose positive-class probability variation across factorial universes.

    ``design`` has one row per universe and one column per Choice factor. Its
    index must exactly match the probability columns. Repeated or missing cells
    are rejected. All universes have equal weight. The residual aggregates all
    interaction orders; it does not identify any particular factor pairing.
    """
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or 0 in values.shape or not np.isfinite(values).all():
        raise ValueError("Attribution values must be a finite nonempty matrix")
    if not probabilities.columns.equals(design.index):
        raise ValueError("Probability columns must match the design index in order")
    if not design.index.is_unique or not design.columns.is_unique:
        raise ValueError("Design universe IDs and factor names must be unique")
    if INTERACTION in design.columns:
        raise ValueError(f"{INTERACTION!r} is reserved for the interaction residual")
    if design.isna().any().any():
        raise ValueError("Factor levels must be nonmissing")
    levels = {factor: design[factor].unique() for factor in design}
    expected = prod(len(options) for options in levels.values())
    if len(design) != expected or (len(levels) > 0 and design.duplicated().any()):
        raise ValueError("Attribution requires a complete balanced factorial design")

    differences = values - values[:, :1]
    grand_difference = differences.mean(axis=1)
    centered = differences - grand_difference[:, None]
    total = np.sum(centered**2, axis=1)
    effects: dict[str, np.ndarray] = {}
    means = []
    for factor, options in levels.items():
        effect = np.zeros(len(values))
        for level in options:
            mask = (design[factor] == level).to_numpy()
            count = int(mask.sum())
            level_difference = differences[:, mask].mean(axis=1)
            effect += count * (level_difference - grand_difference) ** 2
            means.append(
                pd.DataFrame(
                    {
                        "sample_index": probabilities.index,
                        "factor": factor,
                        "level": level,
                        "mean_probability": values[:, mask].mean(axis=1),
                        "universe_count": count,
                    }
                )
            )
        effects[factor] = effect
    for a, b in combinations(levels, 2):
        effect = np.zeros(len(values))
        for level_a in levels[a]:
            mask_a = (design[a] == level_a).to_numpy()
            mean_a = differences[:, mask_a].mean(axis=1)
            for level_b in levels[b]:
                mask_b = (design[b] == level_b).to_numpy()
                mean_b = differences[:, mask_b].mean(axis=1)
                mask = mask_a & mask_b
                cell = differences[:, mask].mean(axis=1)
                effect += (
                    int(mask.sum()) * (cell - mean_a - mean_b + grand_difference) ** 2
                )
        effects[f"{a} × {b}"] = effect
    effects[INTERACTION] = np.maximum(0, total - sum(effects.values()))
    ss = pd.DataFrame(effects, index=probabilities.index)
    contributions = ss.div(np.where(total > 0, total, 1), axis=0)
    return Attribution(
        contributions=contributions,
        sums_of_squares=ss,
        total_sum_of_squares=pd.Series(
            total, index=probabilities.index, name="total_ss"
        ),
        level_means=pd.concat(means, ignore_index=True)
        if means
        else pd.DataFrame(
            columns=[
                "sample_index",
                "factor",
                "level",
                "mean_probability",
                "universe_count",
            ]
        ),
        perfectly_stable=pd.Series(
            (values == values[:, :1]).all(axis=1),
            index=probabilities.index,
            name="perfectly_stable",
        ),
    )

"""Descriptive metrics over equally weighted preprocessing universes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


def validate_probabilities(probabilities: ArrayLike) -> NDArray[np.float64]:
    """Validate a nonempty sample-by-universe matrix of probabilities."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("Probabilities must be a nonempty sample-by-universe matrix")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilities must be finite and between zero and one")
    return values


def prediction_metrics(
    probabilities: ArrayLike, *, threshold: float = 0.5
) -> pd.DataFrame:
    """Summarize each row; column zero is the nominal universe.

    Universes have equal weight. Standard deviation uses ddof=0. A probability
    equal to the threshold predicts the positive class. Class flips count
    universes disagreeing with the nominal decision. Decision entropy is in
    bits and concerns class votes, not the mean model probability. Ties select
    the positive majority decision, with agreement 0.5.
    """
    values = validate_probabilities(probabilities)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between zero and one")
    decisions = values >= threshold
    positive_count = decisions.sum(axis=1)
    fraction = positive_count / values.shape[1]
    entropy = np.zeros(len(values))
    for part in (fraction, 1 - fraction):
        mask = part > 0
        entropy[mask] -= part[mask] * np.log2(part[mask])
    # Center small differences, avoiding cancellation near constant predictions.
    differences = values - values[:, :1]
    centered = differences - differences.mean(axis=1, keepdims=True)
    index = probabilities.index if isinstance(probabilities, pd.DataFrame) else None
    return pd.DataFrame(
        {
            "nominal_probability": values[:, 0],
            "mean_probability": values.mean(axis=1),
            "standard_deviation": np.sqrt(np.mean(centered**2, axis=1)),
            "minimum_probability": values.min(axis=1),
            "maximum_probability": values.max(axis=1),
            "probability_range": np.ptp(values, axis=1),
            "positive_prediction_count": positive_count,
            "negative_prediction_count": values.shape[1] - positive_count,
            "positive_prediction_fraction": fraction,
            "negative_prediction_fraction": 1 - fraction,
            "decision_agreement": np.maximum(fraction, 1 - fraction),
            "class_flip_count": (decisions != decisions[:, :1]).sum(axis=1),
            "decision_entropy": entropy,
            "nominal_decision": decisions[:, 0].astype(int),
            "majority_decision": (fraction >= 0.5).astype(int),
            "perfectly_stable": (values == values[:, :1]).all(axis=1),
        },
        index=index,
    )

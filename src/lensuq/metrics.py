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
    negative_count = values.shape[1] - positive_count
    fraction = positive_count / values.shape[1]
    negative_fraction = negative_count / values.shape[1]
    entropy = np.zeros(len(values))
    for part in (fraction, negative_fraction):
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
            "negative_prediction_count": negative_count,
            "positive_prediction_fraction": fraction,
            "negative_prediction_fraction": negative_fraction,
            "decision_agreement": np.maximum(positive_count, negative_count)
            / values.shape[1],
            "class_flip_count": (decisions != decisions[:, :1]).sum(axis=1),
            "decision_entropy": entropy,
            "nominal_decision": decisions[:, 0].astype(int),
            "majority_decision": (fraction >= 0.5).astype(int),
            "perfectly_stable": (values == values[:, :1]).all(axis=1),
        },
        index=index,
    )


def multiclass_metrics(
    probabilities: ArrayLike, classes: tuple, index: pd.Index | None = None
) -> pd.DataFrame:
    """Summarize a sample-by-universe-by-class probability tensor."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 3 or 0 in values.shape or values.shape[2] != len(classes):
        raise ValueError(
            "Expected nonempty samples × universes × classes probabilities"
        )
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or not np.allclose(values.sum(axis=2), 1, atol=1e-9)
    ):
        raise ValueError("Invalid multiclass probabilities")
    decisions = values.argmax(axis=2)
    counts = np.stack(
        [(decisions == c).sum(axis=1) for c in range(len(classes))], axis=1
    )
    fractions = counts / values.shape[1]
    majority = counts.argmax(axis=1)
    entropy = -np.sum(
        np.where(
            fractions > 0, fractions * np.log2(np.where(fractions > 0, fractions, 1)), 0
        ),
        axis=1,
    )
    ranges = np.ptp(values, axis=1)
    result = pd.DataFrame(
        {
            "nominal_class": np.asarray(classes)[decisions[:, 0]],
            "majority_class": np.asarray(classes)[majority],
            "decision_agreement": counts.max(axis=1) / values.shape[1],
            "class_flip_count": (decisions != decisions[:, :1]).sum(axis=1),
            "decision_entropy": entropy,
            "probability_range": ranges.max(axis=1),
            "standard_deviation": values.std(axis=1).max(axis=1),
            "perfectly_stable": (values == values[:, :1, :]).all(axis=(1, 2)),
        },
        index=index,
    )
    for c, label in enumerate(classes):
        prefix = f"class_{c}"
        result[f"{prefix}_label"] = label
        result[f"{prefix}_count"] = counts[:, c]
        result[f"{prefix}_frequency"] = fractions[:, c]
        result[f"{prefix}_mean_probability"] = values[:, :, c].mean(axis=1)
        result[f"{prefix}_standard_deviation"] = values[:, :, c].std(axis=1)
        result[f"{prefix}_minimum_probability"] = values[:, :, c].min(axis=1)
        result[f"{prefix}_maximum_probability"] = values[:, :, c].max(axis=1)
        result[f"{prefix}_probability_range"] = ranges[:, c]
    return result


def regression_metrics(
    predictions: ArrayLike, index: pd.Index | None = None
) -> pd.DataFrame:
    """Summarize finite sample-by-universe regression predictions."""
    values = np.asarray(predictions, dtype=float)
    if values.ndim != 2 or 0 in values.shape or not np.isfinite(values).all():
        raise ValueError("Regression predictions must be a finite nonempty matrix")
    return pd.DataFrame(
        {
            "nominal_prediction": values[:, 0],
            "mean_prediction": values.mean(axis=1),
            "standard_deviation": values.std(axis=1),
            "minimum_prediction": values.min(axis=1),
            "maximum_prediction": values.max(axis=1),
            "prediction_range": np.ptp(values, axis=1),
            "median_prediction": np.median(values, axis=1),
            "iqr": np.percentile(values, 75, axis=1)
            - np.percentile(values, 25, axis=1),
            "perfectly_stable": (values == values[:, :1]).all(axis=1),
        },
        index=index,
    )

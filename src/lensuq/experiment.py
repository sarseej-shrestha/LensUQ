"""Independent training and analysis of a preprocessing universe ensemble."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils.validation import check_consistent_length

from lensuq.universe import Universe

if TYPE_CHECKING:
    from lensuq.report import AuditReport
    from lensuq.space import PipelineSpace


class Experiment:
    """Fitted universes sharing training rows, labels, and class ordering.

    Construct through ``PipelineSpace.fit``. Universe pipelines are exposed for
    inspection; mutating or refitting them invalidates the experiment.
    """

    def __init__(self, universes: tuple[Universe, ...], classes: np.ndarray) -> None:
        self.universes = universes
        self.classes = tuple(classes.tolist())
        self._last_report: AuditReport | None = None

    @classmethod
    def _fit(cls, space: PipelineSpace, X: ArrayLike, y: ArrayLike) -> Experiment:
        check_consistent_length(X, y)
        if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
            if not X.index.equals(y.index):
                raise ValueError("X and y indices must match in the same order")
        if type_of_target(y) != "binary" or len(unique_labels(y)) != 2:
            raise ValueError("LensUQ 0.1 requires exactly two target classes")
        classes = unique_labels(y)
        universes = space.universes
        for universe in universes:
            try:
                # Isolate even transformers configured to modify inputs in place.
                universe.pipeline.fit(deepcopy(X), deepcopy(y))
            except ValueError as exc:
                raise ValueError(
                    f"Universe {universe.id} {dict(universe.choices)} "
                    f"failed to fit: {exc}"
                ) from exc
            if not np.array_equal(universe.pipeline.classes_, classes):
                raise ValueError(
                    "All universes must expose the same two target classes"
                )
        return cls(universes, classes)

    @property
    def positive_class(self) -> object:
        """The label in sklearn's second probability column (classes_[1])."""
        return self.classes[1]

    def predict_proba(self, X: ArrayLike) -> pd.DataFrame:
        """Return positive-class probabilities, indexed by sample and universe.

        DataFrame indices must be unique, flat, and nonmissing. Array inputs use
        a RangeIndex. Prediction copies inputs to isolate in-place transformers.
        This method does not change the most recent audit.
        """
        if isinstance(X, pd.DataFrame):
            index = X.index.copy()
            if isinstance(index, pd.MultiIndex):
                raise ValueError("Use a flat sample index, not a MultiIndex")
            if not index.is_unique or index.hasnans:
                raise ValueError("Sample indices must be unique and nonmissing")
        else:
            index = pd.RangeIndex(len(X))
        if len(index) == 0:
            raise ValueError("At least one observation is required")
        columns = {}
        for universe in self.universes:
            values = np.asarray(universe.pipeline.predict_proba(deepcopy(X)))
            if values.shape != (len(index), 2):
                raise ValueError(
                    f"Universe {universe.id} must return two probabilities"
                )
            if (
                not np.isfinite(values).all()
                or (values < 0).any()
                or (values > 1).any()
                or not np.allclose(values.sum(axis=1), 1, rtol=1e-7, atol=1e-9)
            ):
                raise ValueError(
                    f"Universe {universe.id} returned invalid probabilities"
                )
            columns[universe.id] = values[:, 1]
        result = pd.DataFrame(columns, index=index)
        result.columns.name = "universe_id"
        return result

    def audit(self, X: ArrayLike, *, threshold: float = 0.5) -> AuditReport:
        """Analyze observations and retain this report for subsequent explanation."""
        from lensuq.attribution import factorial_attribution
        from lensuq.metrics import prediction_metrics
        from lensuq.report import AuditReport

        probabilities = self.predict_proba(X)
        summary = prediction_metrics(probabilities, threshold=threshold)
        for kind in ("nominal", "majority"):
            summary[f"{kind}_class"] = summary[f"{kind}_decision"].map(
                dict(enumerate(self.classes))
            )
        design = pd.DataFrame(
            [dict(u.choices) for u in self.universes],
            index=pd.Index([u.id for u in self.universes], name="universe_id"),
        )
        attribution = factorial_attribution(probabilities, design)
        rows = []
        for universe in self.universes:
            values = probabilities[universe.id].to_numpy()
            rows.append(
                pd.DataFrame(
                    {
                        "sample_index": probabilities.index,
                        "universe_id": universe.id,
                        **{f"choice:{k}": v for k, v in universe.choices.items()},
                        "probability": values,
                        "predicted_class": np.asarray(self.classes)[
                            (values >= threshold).astype(int)
                        ],
                    }
                )
            )
        samples = pd.DataFrame(X).copy(deep=True)
        samples.index = probabilities.index
        report = AuditReport(
            summary,
            pd.concat(rows, ignore_index=True),
            attribution,
            samples,
            design,
            self.classes,
            threshold,
        )
        self._last_report = report
        return report

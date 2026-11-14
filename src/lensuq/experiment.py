"""Independent fitting and auditing of preprocessing universes."""

from __future__ import annotations

from collections.abc import Hashable
from copy import deepcopy
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import is_classifier
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils.validation import check_consistent_length

from lensuq.universe import Universe

if TYPE_CHECKING:
    from lensuq.report import AuditReport, SampleReport
    from lensuq.space import PipelineSpace


class Experiment:
    """Fitted universes and reproducible execution metadata."""

    def __init__(
        self, universes: tuple[Universe, ...], classes: tuple, task: str, metadata: dict
    ) -> None:
        self.universes = universes
        self.classes = classes
        self.task = task
        self.metadata = metadata
        self._last_report: AuditReport | None = None

    @classmethod
    def _fit(
        cls,
        space: PipelineSpace,
        X: ArrayLike,
        y: ArrayLike,
        *,
        strategy: str = "full",
        n_universes: int | None = None,
        random_state: int | None = None,
        cache: bool = False,
    ) -> Experiment:
        check_consistent_length(X, y)
        if (
            isinstance(X, pd.DataFrame)
            and isinstance(y, pd.Series)
            and not X.index.equals(y.index)
        ):
            raise ValueError("X and y indices must match in the same order")
        task = (
            "classification" if is_classifier(space._stages["model"]) else "regression"
        )
        target = type_of_target(y)
        if task == "classification":
            if target not in {"binary", "multiclass"} or len(unique_labels(y)) < 2:
                raise ValueError(
                    "Classification requires at least two discrete target classes"
                )
            classes = tuple(unique_labels(y).tolist())
        else:
            if target not in {"continuous", "multiclass", "binary"}:
                raise ValueError("Regression requires numeric targets")
            classes = ()
        universes = space.select_universes(
            strategy=strategy, n_universes=n_universes, random_state=random_state
        )
        prefix_cache: dict[tuple, tuple[ArrayLike, tuple]] = {}
        hits = 0
        started = perf_counter()
        for universe in universes:
            try:
                if cache:
                    transformed = deepcopy(X)
                    key: tuple = ()
                    for step_index, (name, template) in enumerate(
                        universe.pipeline.steps[:-1]
                    ):
                        key += ((name, universe.choices.get(name, "__fixed__")),)
                        if key in prefix_cache:
                            hits += 1
                            transformed = deepcopy(prefix_cache[key][0])
                            universe.pipeline.steps[step_index] = (
                                name,
                                deepcopy(prefix_cache[key][1]),
                            )
                            continue
                        if template == "passthrough":
                            fitted = "passthrough"
                        else:
                            fitted = template
                            transformed = fitted.fit_transform(
                                deepcopy(transformed), deepcopy(y)
                            )
                        universe.pipeline.steps[step_index] = (name, fitted)
                        prefix_cache[key] = (deepcopy(transformed), deepcopy(fitted))
                    universe.pipeline.steps[-1][1].fit(
                        deepcopy(transformed), deepcopy(y)
                    )
                else:
                    universe.pipeline.fit(deepcopy(X), deepcopy(y))
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Universe {universe.id} {dict(universe.choices)} "
                    f"failed to fit: {exc}"
                ) from exc
            if task == "classification" and not np.array_equal(
                universe.pipeline.classes_, classes
            ):
                raise ValueError("All universes must expose the same target classes")
        metadata = {
            "task_type": task,
            "factors": {k: list(v) for k, v in space.factors.items()},
            "possible_universes": space.n_universes,
            "executed_universes": len(universes),
            "universe_ids": [u.id for u in universes],
            "strategy": strategy,
            "random_state": random_state,
            "cache_enabled": cache,
            "cache_hits": hits,
            "training_seconds": perf_counter() - started,
            "estimator_class": type(space._stages["model"]).__name__,
        }
        return cls(universes, classes, task, metadata)

    @property
    def positive_class(self) -> object:
        """Second class label for binary classifiers."""
        if self.task != "classification" or len(self.classes) != 2:
            raise ValueError("positive_class is defined only for binary classification")
        return self.classes[1]

    def _index(self, X: ArrayLike) -> pd.Index:
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
        return index

    def _probability_tensor(self, X: ArrayLike) -> tuple[np.ndarray, pd.Index]:
        if self.task != "classification":
            raise ValueError("predict_proba requires a classification experiment")
        index = self._index(X)
        matrices = []
        for universe in self.universes:
            values = np.asarray(
                universe.pipeline.predict_proba(deepcopy(X)), dtype=float
            )
            if (
                values.shape != (len(index), len(self.classes))
                or not np.isfinite(values).all()
                or (values < 0).any()
                or (values > 1).any()
                or not np.allclose(values.sum(axis=1), 1, atol=1e-9)
            ):
                raise ValueError(
                    f"Universe {universe.id} returned invalid probabilities"
                )
            matrices.append(values)
        return np.stack(matrices, axis=1), index

    def predict_proba(self, X: ArrayLike) -> pd.DataFrame | np.ndarray:
        """Binary positive-class frame or multiclass probability tensor."""
        values, index = self._probability_tensor(X)
        if len(self.classes) > 2:
            return values
        result = pd.DataFrame(
            values[:, :, 1], index=index, columns=[u.id for u in self.universes]
        )
        result.columns.name = "universe_id"
        return result

    def predict(self, X: ArrayLike) -> pd.DataFrame:
        """Return one regression prediction per universe and observation."""
        if self.task != "regression":
            raise ValueError("predict requires a regression experiment")
        index = self._index(X)
        values = []
        for universe in self.universes:
            column = np.asarray(universe.pipeline.predict(deepcopy(X)), dtype=float)
            if column.shape != (len(index),) or not np.isfinite(column).all():
                raise ValueError(f"Universe {universe.id} returned invalid predictions")
            values.append(column)
        return pd.DataFrame(
            np.stack(values, axis=1),
            index=index,
            columns=[u.id for u in self.universes],
        )

    def audit(
        self, X: ArrayLike, *, threshold: float = 0.5, class_index: int | None = None
    ) -> AuditReport:
        """Analyze predictions. Attribution requires a complete factorial design."""
        from lensuq.attribution import factorial_attribution
        from lensuq.metrics import (
            multiclass_metrics,
            prediction_metrics,
            regression_metrics,
        )
        from lensuq.report import AuditReport

        started = perf_counter()
        if self.task == "classification":
            tensor, index = self._probability_tensor(X)
            if len(self.classes) == 2:
                probabilities = pd.DataFrame(
                    tensor[:, :, 1], index=index, columns=[u.id for u in self.universes]
                )
                summary = prediction_metrics(probabilities, threshold=threshold)
                for kind in ("nominal", "majority"):
                    summary[f"{kind}_class"] = summary[f"{kind}_decision"].map(
                        dict(enumerate(self.classes))
                    )
                selected = probabilities
                chosen_class = 1
                decisions = (tensor[:, :, 1] >= threshold).astype(int)
            else:
                summary = multiclass_metrics(tensor, self.classes, index)
                if class_index is not None and (
                    not isinstance(class_index, int)
                    or not 0 <= class_index < len(self.classes)
                ):
                    raise ValueError("class_index is outside the class range")
                chosen_class = class_index if class_index is not None else None
                decisions = tensor.argmax(axis=2)
                selected = None
        else:
            values = self.predict(X)
            index = values.index
            summary = regression_metrics(values, index)
            selected = values
            chosen_class = None
            decisions = None
        design = pd.DataFrame(
            [dict(u.choices) for u in self.universes],
            index=pd.Index([u.id for u in self.universes], name="universe_id"),
        )
        if self.metadata["strategy"] == "full":
            if selected is not None:
                attribution = factorial_attribution(selected, design)
            else:
                # Decompose each row's selected class probability.
                selected_classes = (
                    np.full(len(index), chosen_class)
                    if chosen_class is not None
                    else summary["majority_class"]
                    .map({label: i for i, label in enumerate(self.classes)})
                    .to_numpy()
                )
                selected_values = tensor[
                    np.arange(len(index))[:, None],
                    np.arange(len(self.universes))[None, :],
                    selected_classes[:, None],
                ]
                selected = pd.DataFrame(
                    selected_values, index=index, columns=design.index
                )
                attribution = factorial_attribution(selected, design)
        else:
            attribution = None
        rows = []
        for j, universe in enumerate(self.universes):
            row = {
                "sample_index": index,
                "universe_id": universe.id,
                **{f"choice:{k}": v for k, v in universe.choices.items()},
            }
            if self.task == "classification":
                if len(self.classes) == 2:
                    row["probability"] = tensor[:, j, 1]
                else:
                    row["winning_probability"] = tensor[:, j, :].max(axis=1)
                row["predicted_class"] = np.asarray(self.classes)[decisions[:, j]]
                if len(self.classes) > 2:
                    for c in range(len(self.classes)):
                        row[f"class_{c}_probability"] = tensor[:, j, c]
            else:
                row["prediction"] = values.iloc[:, j].to_numpy()
            rows.append(pd.DataFrame(row))
        samples = pd.DataFrame(X).copy(deep=True)
        samples.index = index
        metadata = dict(
            self.metadata,
            threshold=threshold
            if self.task == "classification" and len(self.classes) == 2
            else None,
            classes=list(self.classes),
            attribution_class_index=chosen_class,
            analysis_seconds=perf_counter() - started,
        )
        report = AuditReport(
            summary,
            pd.concat(rows, ignore_index=True),
            attribution,
            samples,
            design,
            self.classes,
            threshold,
            metadata,
        )
        self._last_report = report
        return report

    def most_sensitive(self, n: int = 10) -> pd.DataFrame:
        """Rank samples from the latest successful audit."""
        return self._require_report().most_sensitive(n)

    def _require_report(self) -> AuditReport:
        if self._last_report is None:
            raise RuntimeError("Call audit(X) before ranking or explaining samples")
        return self._last_report

    def explain(self, sample_index: Hashable) -> SampleReport:
        """Explain an index label from the latest successful audit."""
        return self._require_report().sample(sample_index)

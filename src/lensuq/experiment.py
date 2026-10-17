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
    from lensuq.space import PipelineSpace


class Experiment:
    """Fitted universes sharing training rows, labels, and class ordering.

    Construct through ``PipelineSpace.fit``. Universe pipelines are exposed for
    inspection; mutating or refitting them invalidates the experiment.
    """

    def __init__(self, universes: tuple[Universe, ...], classes: np.ndarray) -> None:
        self.universes = universes
        self.classes = tuple(classes.tolist())

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
                    f"Universe {universe.id} {dict(universe.choices)} failed to fit: {exc}"
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

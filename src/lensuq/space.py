"""Complete discrete factorial spaces for ordered sklearn pipelines."""

from __future__ import annotations

from itertools import product
from math import prod

import pandas as pd
from sklearn.base import BaseEstimator, clone, is_classifier

from lensuq.choice import Choice
from lensuq.universe import Universe


class PipelineSpace:
    """An ordered preprocessing chain ending in a fixed binary classifier.

    Keyword order defines execution order. The final keyword must be ``model``;
    all preceding stages must implement fit/transform. Only preprocessing stages
    may be Choices. The first Cartesian combination is nominal, never selected
    using performance. Each access to ``universes`` builds fresh clones.
    """

    def __init__(self, **stages: BaseEstimator | Choice) -> None:
        if not stages or list(stages)[-1] != "model":
            raise ValueError("The final stage must be named 'model'")
        self._stages: dict[str, BaseEstimator | Choice] = {}
        for name, stage in stages.items():
            if not name or "__" in name or name in Pipeline_parameters:
                raise ValueError(f"Invalid pipeline stage name: {name!r}")
            if name == "model":
                if isinstance(stage, Choice):
                    raise ValueError("The model must be fixed, not a Choice")
                if not isinstance(stage, BaseEstimator) or not is_classifier(stage):
                    raise TypeError("model must be a scikit-learn classifier")
                if not callable(getattr(stage, "predict_proba", None)):
                    raise TypeError("model must implement predict_proba")
            else:
                options = stage.options if isinstance(stage, Choice) else (stage,)
                for option in options:
                    if not isinstance(option, BaseEstimator) or not all(
                        callable(getattr(option, method, None))
                        for method in ("fit", "transform")
                    ):
                        raise TypeError(f"Stage {name!r} must contain transformers")
            self._stages[name] = (
                Choice(*stage.options, name=stage.name, labels=stage.labels)
                if isinstance(stage, Choice)
                else clone(stage)
            )

    @property
    def factors(self) -> dict[str, tuple[str, ...]]:
        """Factor names and ordered level labels (including singleton choices)."""
        return {
            name: stage.labels
            for name, stage in self._stages.items()
            if isinstance(stage, Choice)
        }

    @property
    def n_universes(self) -> int:
        """Size of the complete Cartesian product."""
        return prod(len(levels) for levels in self.factors.values())

    @property
    def universes(self) -> tuple[Universe, ...]:
        """Fresh universes; the rightmost factor varies fastest."""
        alternatives = [
            list(zip(stage.labels, stage.options, strict=True))
            if isinstance(stage, Choice)
            else [(None, stage)]
            for stage in self._stages.values()
        ]
        result = []
        for id, combination in enumerate(product(*alternatives)):
            choices, steps = {}, []
            for name, (label, estimator) in zip(self._stages, combination, strict=True):
                steps.append((name, estimator))
                if label is not None:
                    choices[name] = label
            result.append(Universe.from_steps(id, choices, steps))
        return tuple(result)

    @property
    def metadata(self) -> pd.DataFrame:
        """Universe coordinates without estimator objects."""
        return pd.DataFrame(
            [
                {"universe_id": id, **dict(zip(self.factors, levels, strict=True))}
                for id, levels in enumerate(product(*self.factors.values()))
            ]
        ).set_index("universe_id")

    def __repr__(self) -> str:
        return (
            f"PipelineSpace(n_universes={self.n_universes}, factors={self.factors!r})"
        )


# sklearn reserves these names for Pipeline constructor parameters.
Pipeline_parameters = {"steps", "memory", "verbose", "transform_input"}

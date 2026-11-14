"""Ordered, named alternatives for a preprocessing stage."""

from __future__ import annotations

import re
from collections.abc import Sequence
from itertools import product
from typing import Generic, TypeVar

from sklearn.base import BaseEstimator, clone

Estimator = TypeVar("Estimator", bound=BaseEstimator)


class Choice(Generic[Estimator]):
    """Snapshot ordered estimator alternatives without fitting or mutating them.

    ``name`` is an optional descriptive name; the PipelineSpace keyword names
    the factor. Explicit ``labels`` must be unique, nonempty strings. Otherwise
    labels use estimator class names, strategy names, and collision suffixes.
    """

    def __init__(
        self,
        *options: Estimator | str,
        name: str | None = None,
        labels: Sequence[str] | None = None,
    ) -> None:
        if not options:
            raise ValueError("Choice requires at least one option")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ValueError("Choice name must be a nonempty string")
        if not all(
            isinstance(option, BaseEstimator) or option == "passthrough"
            for option in options
        ):
            raise TypeError(
                "Choice options must be scikit-learn estimators or 'passthrough'"
            )
        self.name = name
        self._options = tuple(
            clone(option) if isinstance(option, BaseEstimator) else option
            for option in options
        )
        if labels is None:
            generated: list[str] = []
            for option in options:
                base = (
                    re.sub(r"(?<!^)(?=[A-Z][a-z])", "_", type(option).__name__)
                    if option != "passthrough"
                    else "passthrough"
                )
                base = base.lower()
                strategy = (
                    option.get_params(deep=False).get("strategy")
                    if isinstance(option, BaseEstimator)
                    else None
                )
                if isinstance(strategy, str):
                    base += f"_{strategy}"
                label, suffix = base, 2
                while label in generated:
                    label = f"{base}_{suffix}"
                    suffix += 1
                generated.append(label)
            labels = generated
        if isinstance(labels, str) or len(labels) != len(options):
            raise ValueError("Provide one label per option")
        if any(not isinstance(label, str) or not label.strip() for label in labels):
            raise ValueError("Option labels must be nonempty strings")
        if len(set(labels)) != len(labels):
            raise ValueError("Option labels must be unique")
        self._labels = tuple(labels)

    @property
    def labels(self) -> tuple[str, ...]:
        """Deterministic option identifiers in declaration order."""
        return self._labels

    @property
    def options(self) -> tuple[Estimator | str, ...]:
        """Fresh unfitted clones, so callers cannot mutate the stored templates."""
        return tuple(
            clone(option) if isinstance(option, BaseEstimator) else option
            for option in self._options
        )

    def __len__(self) -> int:
        return len(self._options)

    def __repr__(self) -> str:
        return f"Choice(name={self.name!r}, labels={self.labels!r})"


class ParamChoice(Choice):
    """Finite Cartesian alternatives for an estimator class and parameter values."""

    def __init__(
        self, estimator_type: type[BaseEstimator], **parameters: Sequence
    ) -> None:
        if not isinstance(estimator_type, type) or not issubclass(
            estimator_type, BaseEstimator
        ):
            raise TypeError("estimator_type must be a scikit-learn estimator class")
        if not parameters:
            raise ValueError("ParamChoice requires at least one parameter")
        for name, values in parameters.items():
            if (
                isinstance(values, (str, bytes))
                or not isinstance(values, Sequence)
                or not values
            ):
                raise ValueError(f"Parameter {name!r} requires a nonempty sequence")
        keys = tuple(parameters)
        options = []
        labels = []
        for values in product(*(parameters[key] for key in keys)):
            kwargs = dict(zip(keys, values, strict=True))
            options.append(estimator_type(**kwargs))
            parts = []
            for key, value in kwargs.items():
                label_value = (
                    value.__name__
                    if callable(value) and hasattr(value, "__name__")
                    else repr(value)
                )
                parts.append(f"{key}={label_value}")
            labels.append(",".join(parts))
        super().__init__(*options, labels=labels)

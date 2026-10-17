"""Ordered, named alternatives for a preprocessing stage."""

from __future__ import annotations

import re
from collections.abc import Sequence
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
        *options: Estimator,
        name: str | None = None,
        labels: Sequence[str] | None = None,
    ) -> None:
        if not options:
            raise ValueError("Choice requires at least one option")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ValueError("Choice name must be a nonempty string")
        if not all(isinstance(option, BaseEstimator) for option in options):
            raise TypeError("Choice options must be scikit-learn estimators")
        self.name = name
        self._options = tuple(clone(option) for option in options)
        if labels is None:
            generated: list[str] = []
            for option in options:
                base = re.sub(r"(?<!^)(?=[A-Z][a-z])", "_", type(option).__name__)
                base = base.lower()
                strategy = option.get_params(deep=False).get("strategy")
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
    def options(self) -> tuple[Estimator, ...]:
        """Fresh unfitted clones, so callers cannot mutate the stored templates."""
        return tuple(clone(option) for option in self._options)

    def __len__(self) -> int:
        return len(self._options)

    def __repr__(self) -> str:
        return f"Choice(name={self.name!r}, labels={self.labels!r})"

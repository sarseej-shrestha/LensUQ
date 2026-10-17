"""One concrete, independently fitted preprocessing configuration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class Universe:
    """A pipeline and its factorial coordinates; ID zero is nominal.

    The pipeline is mutable because fitting updates its estimators. Construct
    universes through ``from_steps`` to guarantee independent estimator clones.
    """

    id: int
    choices: Mapping[str, str]
    pipeline: Pipeline

    def __post_init__(self) -> None:
        object.__setattr__(self, "choices", MappingProxyType(dict(self.choices)))

    @classmethod
    def from_steps(
        cls,
        id: int,
        choices: Mapping[str, str],
        steps: Sequence[tuple[str, BaseEstimator]],
    ) -> Universe:
        """Build a fresh pipeline, discarding any supplied fitted state."""
        return cls(id, choices, Pipeline([(name, clone(step)) for name, step in steps]))

    @property
    def is_nominal(self) -> bool:
        """Whether this is the first declared combination."""
        return self.id == 0

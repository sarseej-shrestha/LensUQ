"""Structured audit snapshots with optional Rich terminal rendering."""

from __future__ import annotations

import pandas as pd
from rich.console import Console, Group
from rich.table import Table
from rich.text import Text

from lensuq.attribution import Attribution


class AuditReport:
    """An audit snapshot; tabular properties return independent copies.

    Use ``report.dataframe.sort_values(...)`` or ``.loc[...]`` for arbitrary
    sorting and filtering. Long-form predictions retain every universe's factor
    coordinates, positive-class probability, and thresholded class label.
    """

    def __init__(
        self,
        summary: pd.DataFrame,
        predictions: pd.DataFrame,
        attribution: Attribution,
        samples: pd.DataFrame,
        design: pd.DataFrame,
        classes: tuple[object, object],
        threshold: float,
    ) -> None:
        self._summary = summary.copy(deep=True)
        self._predictions = predictions.copy(deep=True)
        self._attribution = attribution
        self._samples = samples.copy(deep=True)
        self._design = design.copy(deep=True)
        self.classes = classes
        self.threshold = threshold

    @property
    def dataframe(self) -> pd.DataFrame:
        """One row per audited observation, retaining its original index."""
        return self._summary.copy(deep=True)

    @property
    def predictions(self) -> pd.DataFrame:
        """One row per sample and universe; factor columns use 'choice:' prefixes."""
        return self._predictions.copy(deep=True)

    @property
    def contributions(self) -> pd.DataFrame:
        """Main-effect and aggregate interaction fractions for each sample."""
        return self._attribution.contributions.copy(deep=True)

    @property
    def level_means(self) -> pd.DataFrame:
        """Per-sample factor-level mean probabilities and universe counts."""
        return self._attribution.level_means.copy(deep=True)

    @property
    def n_universes(self) -> int:
        """Number of equally weighted universes in this audit."""
        return len(self._design)

    def __rich__(self) -> Group:
        overview = Table.grid(padding=(0, 2))
        overview.add_row("Observations", str(len(self._summary)))
        overview.add_row("Pipeline universes", str(self.n_universes))
        overview.add_row("Positive class", str(self.classes[1]))
        overview.add_row("Decision threshold", f"{self.threshold:g}")
        overview.add_row(
            "Observations with class disagreement",
            str(int((self._summary.decision_agreement < 1).sum())),
        )
        return Group(
            Text("LensUQ Prediction Audit", style="bold cyan"),
            overview,
            Text(
                "Variation is conditional on the declared preprocessing alternatives; "
                "it is not a calibrated uncertainty interval."
            ),
        )

    def render(self, console: Console | None = None) -> None:
        """Print a concise terminal summary without changing the report."""
        (console or Console()).print(self)

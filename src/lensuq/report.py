"""Structured audit snapshots with optional Rich terminal rendering."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from numbers import Integral

import pandas as pd
from rich.console import Console, Group
from rich.table import Table
from rich.text import Text

from lensuq.attribution import Attribution


@dataclass(frozen=True)
class SampleReport:
    """A standalone explanation for one audited observation."""

    sample_index: Hashable
    sample: pd.Series
    metrics: pd.Series
    nominal_choices: dict[str, str]
    predictions: pd.DataFrame
    contributions: pd.Series
    sums_of_squares: pd.Series
    total_sum_of_squares: float
    level_means: pd.DataFrame
    positive_class: object
    threshold: float

    def __rich__(self) -> Group:
        m = self.metrics
        overview = Table.grid(padding=(0, 2))
        for label, value in (
            (
                "Positive class / threshold",
                f"{self.positive_class} / {self.threshold:g}",
            ),
            (
                "Nominal choices",
                ", ".join(f"{k}={v}" for k, v in self.nominal_choices.items())
                or "Fixed pipeline",
            ),
            (
                "Nominal probability / class",
                f"{m.nominal_probability:.6f} / {m.nominal_class}",
            ),
            (
                "Mean / population std",
                f"{m.mean_probability:.6f} / {m.standard_deviation:.6f}",
            ),
            (
                "Probability min – max",
                f"{m.minimum_probability:.6f} – {m.maximum_probability:.6f}",
            ),
            ("Majority class", str(m.majority_class)),
            ("Decision agreement", f"{m.decision_agreement:.2%}"),
            ("Class flips vs nominal", str(m.class_flip_count)),
            ("Decision entropy (bits)", f"{m.decision_entropy:.6f}"),
        ):
            overview.add_row(label, value)
        predictions = Table(title="All universe predictions")
        for column in self.predictions:
            predictions.add_column(column.removeprefix("choice:"))
        for row in self.predictions.itertuples(index=False, name=None):
            predictions.add_row(
                *[
                    f"{v:.6f}" if col == "probability" else str(v)
                    for col, v in zip(self.predictions.columns, row, strict=True)
                ]
            )
        attribution = Table(title="Probability variation attribution")
        attribution.add_column("Factor")
        attribution.add_column("Contribution", justify="right")
        for name, contribution in self.contributions.items():
            attribution.add_row(str(name), f"{contribution:.2%}")
        levels = Table(title="Factor-level mean probabilities")
        for label in ("Factor", "Level", "Mean", "Universes"):
            levels.add_column(label)
        for row in self.level_means.itertuples(index=False):
            levels.add_row(
                str(row.factor),
                str(row.level),
                f"{row.mean_probability:.6f}",
                str(row.universe_count),
            )
        if m.perfectly_stable:
            interpretation = (
                "All selected universes produced exactly the same probability."
            )
        elif m.decision_agreement < 1:
            interpretation = (
                "This observation's predicted class changes across the selected "
                "preprocessing configurations."
            )
        else:
            interpretation = (
                "Probabilities vary across the selected configurations, while all "
                "universes agree on the class at this threshold."
            )
        return Group(
            Text(f"Sample {self.sample_index!r}", style="bold cyan"),
            overview,
            predictions,
            attribution,
            levels,
            Text(interpretation),
            Text(
                "Interaction residual aggregates interactions; "
                "it does not identify specific pairs."
            ),
        )

    def render(self, console: Console | None = None) -> None:
        """Print metrics, every universe prediction, attribution, and level means."""
        (console or Console()).print(self)


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

    def most_sensitive(self, n: int = 10) -> pd.DataFrame:
        """Rank by agreement ascending, range descending, then std descending.

        Exact ties retain audit input order. ``n=0`` returns an empty table.
        """
        if isinstance(n, bool) or not isinstance(n, Integral) or n < 0:
            raise ValueError("n must be a nonnegative integer")
        ranked = self._summary.sort_values(
            ["decision_agreement", "probability_range", "standard_deviation"],
            ascending=[True, False, False],
            kind="stable",
        )
        return ranked.head(n).copy(deep=True)

    def sample(self, sample_index: Hashable) -> SampleReport:
        """Explain a sample by index label, never by its row position."""
        if sample_index not in self._summary.index:
            raise KeyError(f"Sample {sample_index!r} is not in this audit")
        predictions = self._predictions.loc[
            self._predictions.sample_index == sample_index
        ].drop(columns="sample_index")
        levels = self._attribution.level_means
        return SampleReport(
            sample_index=sample_index,
            sample=self._samples.loc[sample_index].copy(deep=True),
            metrics=self._summary.loc[sample_index].copy(deep=True),
            nominal_choices=self._design.iloc[0].to_dict(),
            predictions=predictions.reset_index(drop=True),
            contributions=self.contributions.loc[sample_index].copy(deep=True),
            sums_of_squares=self._attribution.sums_of_squares.loc[sample_index].copy(
                deep=True
            ),
            total_sum_of_squares=float(
                self._attribution.total_sum_of_squares.loc[sample_index]
            ),
            level_means=levels.loc[levels.sample_index == sample_index].copy(deep=True),
            positive_class=self.classes[1],
            threshold=self.threshold,
        )

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

"""Structured audit snapshots with optional Rich terminal rendering."""

from __future__ import annotations

import json
from collections.abc import Hashable
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
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
    task: str = "classification"

    def __rich__(self) -> Group:
        m = self.metrics
        overview = Table.grid(padding=(0, 2))
        details = (
            (
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
            )
            if "nominal_probability" in m
            else (
                ("Nominal prediction", f"{m.nominal_prediction:.6f}"),
                (
                    "Mean / population std",
                    f"{m.mean_prediction:.6f} / {m.standard_deviation:.6f}",
                ),
                ("Prediction range", f"{m.prediction_range:.6f}"),
                ("Median / IQR", f"{m.median_prediction:.6f} / {m.iqr:.6f}"),
            )
            if self.task == "regression"
            else (
                ("Nominal / majority class", f"{m.nominal_class} / {m.majority_class}"),
                ("Decision agreement", f"{m.decision_agreement:.2%}"),
                ("Decision entropy (bits)", f"{m.decision_entropy:.6f}"),
            )
        )
        for label, value in details:
            overview.add_row(label, value)
        predictions = Table(title="All universe predictions")
        for column in self.predictions:
            predictions.add_column(column.removeprefix("choice:"))
        for row in self.predictions.itertuples(index=False, name=None):
            predictions.add_row(
                *[
                    f"{v:.6f}"
                    if col == "probability"
                    or col == "prediction"
                    or col.endswith("_probability")
                    else str(v)
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
        elif self.task == "regression":
            interpretation = "Predictions vary across the selected configurations."
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
            Text(f"Sample {self.sample_index}", style="bold cyan"),
            overview,
            predictions,
            attribution,
            levels,
            Text(interpretation),
            Text(
                "Interaction residual contains variation beyond main and "
                "pairwise effects."
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
        attribution: Attribution | None,
        samples: pd.DataFrame,
        design: pd.DataFrame,
        classes: tuple,
        threshold: float,
        metadata: dict | None = None,
    ) -> None:
        self._summary = summary.copy(deep=True)
        self._predictions = predictions.copy(deep=True)
        self._attribution = attribution
        self._samples = samples.copy(deep=True)
        self._design = design.copy(deep=True)
        self.classes = classes
        self.threshold = threshold
        self.metadata = metadata or {"task_type": "classification"}

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
        """Main, pairwise, and residual fractions for full factorial audits."""
        if self._attribution is None:
            raise ValueError(
                "Attribution requires a complete balanced factorial design"
            )
        return self._attribution.contributions.copy(deep=True)

    @property
    def level_means(self) -> pd.DataFrame:
        """Per-sample factor-level mean probabilities and universe counts."""
        if self._attribution is None:
            raise ValueError(
                "Attribution requires a complete balanced factorial design"
            )
        return self._attribution.level_means.copy(deep=True)

    @property
    def summary(self) -> pd.DataFrame:
        """Sample-level metrics."""
        return self.dataframe

    @property
    def samples(self) -> pd.DataFrame:
        """Raw audited observations."""
        return self._samples.copy(deep=True)

    @property
    def universes(self) -> pd.DataFrame:
        """Executed universe coordinates."""
        return self._design.copy(deep=True)

    @property
    def attribution(self) -> Attribution | None:
        """Full-factorial decomposition, unavailable for sampled designs."""
        return self._attribution

    def to_dataframe(self) -> pd.DataFrame:
        """Return sample-level metrics."""
        return self.dataframe

    def to_csv(self, path: str | Path) -> None:
        """Write the sample-level table as CSV."""
        self._summary.to_csv(path, index_label="sample_index")

    def to_json(self, path: str | Path) -> None:
        """Write metadata, universes, metrics, predictions, and attribution."""

        def scalar(value):
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, (pd.Timestamp, pd.Timedelta)):
                return str(value)
            raise TypeError(f"Cannot serialize {type(value).__name__}")

        payload = {
            "metadata": self.metadata,
            "universes": [
                {"universe_id": index, **row.to_dict()}
                for index, row in self._design.iterrows()
            ],
            "samples": [
                {"sample_index": index, **row.to_dict()}
                for index, row in self._summary.iterrows()
            ],
            "predictions": self._predictions.to_dict(orient="records"),
            "attribution": None
            if self._attribution is None
            else [
                {"sample_index": index, **row.to_dict()}
                for index, row in self._attribution.contributions.iterrows()
            ],
        }
        Path(path).write_text(
            json.dumps(payload, default=scalar, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

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
        task = self.metadata.get("task_type", "classification")
        columns = (
            ["prediction_range", "standard_deviation"]
            if task == "regression"
            else ["decision_agreement", "probability_range", "standard_deviation"]
        )
        ranked = self._summary.sort_values(
            columns,
            ascending=[False, False] if task == "regression" else [True, False, False],
            kind="stable",
        )
        return ranked.head(n).copy(deep=True)

    def sample(self, sample_index: Hashable) -> SampleReport:
        """Explain a sample by index label, never by its row position."""
        if sample_index not in self._summary.index:
            raise KeyError(f"Sample {sample_index!r} is not in this audit")
        if self._attribution is None:
            raise ValueError(
                "Attribution requires a complete balanced factorial design"
            )
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
            positive_class=self.classes[1] if len(self.classes) == 2 else None,
            threshold=self.threshold,
            task=self.metadata.get("task_type", "classification"),
        )

    def __rich__(self) -> Group:
        overview = Table.grid(padding=(0, 2))
        overview.add_row("Observations", str(len(self._summary)))
        overview.add_row("Pipeline universes", str(self.n_universes))
        overview.add_row("Task", self.metadata.get("task_type", "classification"))
        if "decision_agreement" in self._summary:
            overview.add_row(
                "Observations with class disagreement",
                str(int((self._summary.decision_agreement < 1).sum())),
            )
        if len(self.classes) == 2:
            overview.add_row("Positive class", str(self.classes[1]))
            overview.add_row("Decision threshold", f"{self.threshold:g}")
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

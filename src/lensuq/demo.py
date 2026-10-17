"""Offline, reproducible demonstration with controlled artificial missingness."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.datasets import load_breast_cancer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from lensuq import Choice, PipelineSpace
from lensuq.experiment import Experiment
from lensuq.report import AuditReport

MISSING_FEATURES = (
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean concavity",
    "mean concave points",
)
DEMO_NOTICE = (
    "Missingness is introduced artificially for the prototype to create a "
    "controlled preprocessing-sensitivity experiment. "
    "The demo is not a medical finding."
)


def introduce_missingness(
    X: pd.DataFrame, *, fraction: float = 0.15, seed: int = 42
) -> pd.DataFrame:
    """Copy X and independently mask selected feature cells at a fixed rate.

    This synthetic mechanism uses neither labels nor fitted data statistics.
    It is a demonstration device, not a model of clinical missingness.
    """
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("fraction must be between zero and one")
    result = X.copy(deep=True)
    mask = (
        np.random.default_rng(seed).random((len(X), len(MISSING_FEATURES))) < fraction
    )
    result.loc[:, list(MISSING_FEATURES)] = result.loc[:, list(MISSING_FEATURES)].mask(
        mask
    )
    return result


@dataclass(frozen=True)
class DemoResult:
    """Reproducible demonstration artifacts for terminal output and tests."""

    space: PipelineSpace
    experiment: Experiment
    report: AuditReport
    training_observations: int
    missing_cells: int
    eligible_cells: int


def build_demo() -> DemoResult:
    """Train nine universes on one stratified split and audit its held-out rows."""
    dataset = load_breast_cancer(as_frame=True)
    X = introduce_missingness(dataset.data)
    X_train, X_test, y_train, _ = train_test_split(
        X, dataset.target, test_size=0.2, stratify=dataset.target, random_state=42
    )
    space = PipelineSpace(
        imputer=Choice(
            SimpleImputer(strategy="mean"),
            SimpleImputer(strategy="median"),
            KNNImputer(n_neighbors=5),
            labels=("mean", "median", "knn_5"),
        ),
        scaler=Choice(
            StandardScaler(),
            RobustScaler(),
            MinMaxScaler(),
            labels=("standard", "robust", "minmax"),
        ),
        model=LogisticRegression(max_iter=5000, random_state=42),
    )
    experiment = space.fit(X_train, y_train)
    return DemoResult(
        space,
        experiment,
        experiment.audit(X_test),
        len(X_train),
        int(X.isna().sum().sum()),
        len(X) * len(MISSING_FEATURES),
    )


def run_demo(console: Console | None = None) -> DemoResult:
    """Print the full demonstration and return its structured results."""
    console = console or Console()
    console.print(DEMO_NOTICE, style="yellow")
    console.print(
        "Traditional view: one preprocessing pipeline -> one model "
        "-> one probability.\n"
        "LensUQ view: same raw observation -> independently trained preprocessing "
        "universes -> prediction distribution -> stability + attribution.\n"
        "No universe is selected using test performance."
    )
    result = build_demo()
    console.print(
        f"\nTrain/test: {result.training_observations}/{len(result.report.dataframe)}; "
        f"artificial missing cells: {result.missing_cells}/{result.eligible_cells} "
        "across six selected features.\n"
        "Probability refers to class 1 (benign); class 0 is malignant. "
        "Labels are unchanged."
    )
    result.report.render(console)
    ranking = result.experiment.most_sensitive(10)
    table = Table(title="10 most preprocessing-sensitive observations")
    for label in ("Sample", "Agreement", "Range", "Std", "Nominal", "Mean", "Flips"):
        table.add_column(label, justify="right")
    for sample_index, row in ranking.iterrows():
        table.add_row(
            str(sample_index),
            f"{row.decision_agreement:.2%}",
            f"{row.probability_range:.6f}",
            f"{row.standard_deviation:.6f}",
            f"{row.nominal_probability:.6f}",
            f"{row.mean_probability:.6f}",
            str(int(row.class_flip_count)),
        )
    console.print(table)
    console.print(
        "Ranking: agreement ascending, range descending, population std descending; "
        "exact ties retain test input order."
    )
    result.experiment.explain(ranking.index[0]).render(console)
    return result

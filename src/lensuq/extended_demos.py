"""Offline multiclass and regression demonstrations."""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from sklearn.datasets import load_diabetes, load_wine
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from lensuq import Choice, ParamChoice, PipelineSpace
from lensuq.experiment import Experiment
from lensuq.report import AuditReport


@dataclass(frozen=True)
class ExtendedDemoResult:
    """Fitted experiment and held-out audit."""

    name: str
    experiment: Experiment
    report: AuditReport


def build_wine_demo() -> ExtendedDemoResult:
    """Audit 3×3 preprocessing choices on the bundled wine data."""
    data = load_wine(as_frame=True)
    train, test, target, _ = train_test_split(
        data.data, data.target, test_size=0.25, stratify=data.target, random_state=42
    )
    space = PipelineSpace(
        scaler=Choice(
            StandardScaler(),
            RobustScaler(),
            MinMaxScaler(),
            labels=("standard", "robust", "minmax"),
        ),
        selector=ParamChoice(SelectKBest, score_func=[f_classif], k=[5, 8, 13]),
        model=LogisticRegression(max_iter=5000, random_state=42),
    )
    experiment = space.fit(train, target)
    return ExtendedDemoResult("wine", experiment, experiment.audit(test))


def build_regression_demo() -> ExtendedDemoResult:
    """Audit 3×3 preprocessing choices on the bundled diabetes data."""
    data = load_diabetes(as_frame=True)
    train, test, target, _ = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42
    )
    space = PipelineSpace(
        scaler=Choice(
            StandardScaler(),
            RobustScaler(),
            MinMaxScaler(),
            labels=("standard", "robust", "minmax"),
        ),
        selector=ParamChoice(SelectKBest, score_func=[f_regression], k=[4, 7, 10]),
        model=Ridge(alpha=1.0),
    )
    experiment = space.fit(train, target)
    return ExtendedDemoResult("regression", experiment, experiment.audit(test))


def run_extended_demo(name: str, console: Console | None = None) -> ExtendedDemoResult:
    """Render actual metrics for a bundled demonstration."""
    result = build_wine_demo() if name == "wine" else build_regression_demo()
    console = console or Console()
    console.print(f"LensUQ {name} demonstration")
    result.report.render(console)
    ranked = result.experiment.most_sensitive(5)
    console.print(ranked.to_string())
    result.experiment.explain(ranked.index[0]).render(console)
    return result

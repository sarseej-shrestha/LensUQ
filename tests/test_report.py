from io import StringIO

import pandas as pd
import pytest
from rich.console import Console
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from lensuq import Choice, PipelineSpace


@pytest.fixture
def audited():
    X = pd.DataFrame({"x": [-3.0, -2, -1, 1, 2, 3]}, index=list("abcdef"))
    experiment = PipelineSpace(
        scale=Choice(StandardScaler(), MinMaxScaler()),
        model=LogisticRegression(random_state=42),
    ).fit(X, [0, 0, 0, 1, 1, 1])
    return experiment, experiment.audit(X), X


def test_report_structure_and_index(audited):
    _, report, X = audited
    assert report.dataframe.index.equals(X.index)
    assert report.dataframe.shape[0] == 6
    assert report.predictions.shape[0] == 12
    assert report.predictions.groupby("sample_index").size().eq(2).all()
    assert set(report.predictions["choice:scale"]) == {
        "standard_scaler",
        "min_max_scaler",
    }
    assert report.contributions.shape == (6, 2)
    assert len(report.level_means) == 12
    assert report.n_universes == 2


def test_report_dataframes_are_independent(audited):
    _, report, _ = audited
    frame = report.dataframe
    frame.loc["a", "mean_probability"] = 42
    assert report.dataframe.loc["a", "mean_probability"] != 42


def test_rich_summary(audited):
    _, report, _ = audited
    output = StringIO()
    report.render(Console(file=output, width=100, color_system=None))
    assert "LensUQ Prediction Audit" in output.getvalue()
    assert "Pipeline universes" in output.getvalue()


def test_audit_class_labels_follow_threshold(audited):
    experiment, _, X = audited
    report = experiment.audit(X, threshold=0)
    assert report.predictions.predicted_class.eq(1).all()
    assert report.dataframe.majority_class.eq(1).all()
    assert report.dataframe.class_flip_count.eq(0).all()

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


def test_ranking_rule_and_exact_ties(audited):
    _, report, _ = audited
    # Deliberately shuffled labels demonstrate that final ties use input order.
    report._summary = pd.DataFrame(
        {
            "decision_agreement": [1, 0.5, 0.5, 0.5, 0.5],
            "probability_range": [0.9, 0.2, 0.3, 0.3, 0.3],
            "standard_deviation": [0.4, 0.1, 0.1, 0.2, 0.2],
        },
        index=["a", "b", "c", "e", "d"],
    )
    assert report.most_sensitive().index.tolist() == ["e", "d", "c", "b", "a"]
    assert report.most_sensitive(0).empty


@pytest.mark.parametrize("n", [-1, 1.5, True])
def test_invalid_ranking_count(audited, n):
    with pytest.raises(ValueError, match="nonnegative integer"):
        audited[1].most_sensitive(n)


def test_experiment_ranks_latest_audit(audited):
    experiment, report, X = audited
    pd.testing.assert_frame_equal(
        experiment.most_sensitive(3), report.most_sensitive(3)
    )
    experiment.audit(X.iloc[:1])
    assert len(experiment.most_sensitive()) == 1
    assert len(report.most_sensitive()) == 6


def test_audit_required_before_ranking():
    experiment = PipelineSpace(model=LogisticRegression()).fit([[0], [1]], [0, 1])
    with pytest.raises(RuntimeError, match="Call audit"):
        experiment.most_sensitive()


def test_sample_explanation_and_rendering(audited):
    experiment, report, X = audited
    sample = experiment.explain("c")
    pd.testing.assert_series_equal(sample.sample, X.loc["c"])
    assert sample.nominal_choices == {"scale": "standard_scaler"}
    assert len(sample.predictions) == 2
    assert len(sample.level_means) == 2
    assert sample.contributions.sum() == pytest.approx(1)
    assert (
        sample.metrics.nominal_probability
        == report.dataframe.loc["c", "nominal_probability"]
    )
    output = StringIO()
    sample.render(Console(file=output, width=120, color_system=None))
    text = output.getvalue()
    for expected in (
        "Sample c",
        "All universe predictions",
        "interaction_residual",
        "Factor-level mean",
    ):
        assert expected in text


def test_missing_sample_is_clear(audited):
    with pytest.raises(KeyError, match="not in this audit"):
        audited[0].explain("missing")


def test_failed_audit_retains_previous_snapshot(audited):
    experiment, report, X = audited
    with pytest.raises(ValueError):
        experiment.audit(X, threshold=-1)
    pd.testing.assert_frame_equal(experiment.most_sensitive(), report.most_sensitive())


def test_fixed_pipeline_explanation_has_zero_contributions():
    experiment = PipelineSpace(model=LogisticRegression()).fit([[0], [1]], [0, 1])
    report = experiment.audit([[0.5]])
    explanation = report.sample(0)
    assert explanation.metrics.perfectly_stable
    assert explanation.contributions.sum() == 0
    assert explanation.nominal_choices == {}
    assert explanation.level_means.empty
    output = StringIO()
    explanation.render(Console(file=output, width=100, color_system=None))
    assert "exactly the same probability" in output.getvalue()


def test_predict_proba_does_not_replace_audit(audited):
    experiment, report, X = audited
    experiment.predict_proba(X.iloc[:1])
    assert len(experiment.most_sensitive()) == len(report.dataframe)

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from lensuq import Choice, PipelineSpace


@pytest.fixture
def training_data():
    X = pd.DataFrame({"a": [0.0, 1, 2, 6, 7, 9], "b": [1.0, 2, 0, 8, 9, 10]})
    return X, pd.Series([0, 0, 0, 1, 1, 1])


def test_independent_fit_without_source_mutation(training_data):
    X, y = training_data
    original = X.copy()
    scaler = StandardScaler(copy=False)
    model = LogisticRegression(random_state=42)
    space = PipelineSpace(scale=Choice(scaler, scaler), model=model)
    experiment = space.fit(X, y)
    assert len(experiment.universes) == 2
    assert experiment.positive_class == 1
    assert not hasattr(scaler, "mean_")
    assert not hasattr(model, "coef_")
    pd.testing.assert_frame_equal(X, original)
    a, b = experiment.universes
    np.testing.assert_allclose(a.pipeline["scale"].mean_, b.pipeline["scale"].mean_)
    assert a.pipeline["model"].coef_ is not b.pipeline["model"].coef_
    assert not hasattr(space.universes[0].pipeline["model"], "coef_")


def test_only_training_data_fits_preprocessors():
    X = [[0.0], [2], [4], [np.nan]]
    experiment = PipelineSpace(
        imputer=Choice(SimpleImputer(), SimpleImputer(strategy="median")),
        model=LogisticRegression(),
    ).fit(X, [0, 0, 1, 1])
    for universe in experiment.universes:
        np.testing.assert_allclose(universe.pipeline["imputer"].statistics_, [2])


@pytest.mark.parametrize("y", [[0] * 6, [0, 1, 2, 0, 1, 2], [0.1] * 6])
def test_binary_targets_required(training_data, y):
    with pytest.raises(ValueError, match="two target classes"):
        PipelineSpace(model=LogisticRegression()).fit(training_data[0], y)


def test_misaligned_targets_rejected(training_data):
    X, y = training_data
    with pytest.raises(ValueError, match="indices"):
        PipelineSpace(model=LogisticRegression()).fit(X, y.iloc[::-1])


def test_fit_error_names_universe():
    with pytest.raises(ValueError, match="Universe 0"):
        PipelineSpace(model=LogisticRegression()).fit([[np.nan], [1]], [0, 1])


def test_probability_matrix_preserves_indices_and_class_labels(training_data):
    X, y = training_data
    experiment = PipelineSpace(
        scale=Choice(StandardScaler(), StandardScaler(with_mean=False)),
        model=LogisticRegression(),
    ).fit(X, y.map({0: "absent", 1: "present"}))
    test = X.iloc[[3, 1]].copy()
    test.index = ["subject-c", "subject-a"]
    result = experiment.predict_proba(test)
    assert result.shape == (2, 2)
    assert result.index.tolist() == ["subject-c", "subject-a"]
    assert result.columns.tolist() == [0, 1]
    assert experiment.positive_class == "present"
    for u in experiment.universes:
        np.testing.assert_allclose(result[u.id], u.pipeline.predict_proba(test)[:, 1])


def test_prediction_does_not_refit_or_mutate_inputs(training_data):
    X, y = training_data
    experiment = PipelineSpace(
        scale=Choice(StandardScaler(copy=False), StandardScaler(copy=False)),
        model=LogisticRegression(),
    ).fit(X, y)
    test = X + 1000
    original = test.copy()
    before = experiment.universes[0].pipeline["scale"].mean_.copy()
    first = experiment.predict_proba(test)
    pd.testing.assert_frame_equal(test, original)
    pd.testing.assert_frame_equal(first, experiment.predict_proba(test))
    np.testing.assert_array_equal(
        before, experiment.universes[0].pipeline["scale"].mean_
    )


@pytest.mark.parametrize("index", [[0, 0], [0, np.nan]])
def test_ambiguous_sample_indices_rejected(training_data, index):
    X, y = training_data
    experiment = PipelineSpace(model=LogisticRegression()).fit(X, y)
    test = X.iloc[:2].copy()
    test.index = index
    with pytest.raises(ValueError, match="unique and nonmissing"):
        experiment.predict_proba(test)


def test_array_prediction_indices(training_data):
    X, y = training_data
    experiment = PipelineSpace(model=LogisticRegression()).fit(X.to_numpy(), y)
    assert experiment.predict_proba(X.to_numpy()).index.equals(pd.RangeIndex(6))
    with pytest.raises(ValueError, match="At least one"):
        experiment.predict_proba(np.empty((0, 2)))

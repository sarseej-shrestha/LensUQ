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

import json
from itertools import product

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, make_regression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from lensuq import Choice, ParamChoice, PipelineSpace
from lensuq.attribution import factorial_attribution
from lensuq.cli import main
from lensuq.extended_demos import build_regression_demo, build_wine_demo
from lensuq.metrics import multiclass_metrics, regression_metrics


def test_parameter_choices_and_passthrough():
    choice = ParamChoice(
        SimpleImputer, strategy=["mean", "median"], add_indicator=[False, True]
    )
    assert len(choice) == 4
    assert choice.labels == (
        "strategy='mean',add_indicator=False",
        "strategy='mean',add_indicator=True",
        "strategy='median',add_indicator=False",
        "strategy='median',add_indicator=True",
    )
    assert Choice("passthrough", StandardScaler()).labels[0] == "passthrough"
    with pytest.raises(ValueError):
        ParamChoice(SimpleImputer, strategy=[])


def test_arbitrary_factors_and_sampling():
    space = PipelineSpace(
        a=Choice(StandardScaler(), MinMaxScaler()),
        b=Choice("passthrough", StandardScaler()),
        c=Choice("passthrough", MinMaxScaler()),
        model=LogisticRegression(),
    )
    assert space.n_universes == 8
    assert [u.id for u in space.universes] == list(range(8))
    a = space.select_universes(strategy="random", n_universes=4, random_state=42)
    b = space.select_universes(strategy="random", n_universes=4, random_state=42)
    assert [u.id for u in a] == [u.id for u in b]
    assert len({u.id for u in a}) == 4
    with pytest.raises(ValueError):
        space.select_universes(strategy="random", n_universes=9)
    with pytest.raises(ValueError):
        space.select_universes(strategy="random", n_universes=0)


def test_multiclass_metrics_hand_computed():
    values = np.array([[[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.8, 0.1]]])
    result = multiclass_metrics(values, ("a", "b", "c"))
    assert result.loc[0, "majority_class"] == "b"
    assert result.loc[0, "decision_agreement"] == pytest.approx(2 / 3)
    assert result.loc[0, "class_1_mean_probability"] == pytest.approx(1.6 / 3)
    assert result.loc[0, "class_1_probability_range"] == pytest.approx(0.7)
    assert result.loc[0, "decision_entropy"] == pytest.approx(0.918295834)


def test_regression_metrics_hand_computed():
    result = regression_metrics([[1, 2, 3, 4], [5, 5, 5, 5]])
    assert result.loc[0, "mean_prediction"] == 2.5
    assert result.loc[0, "median_prediction"] == 2.5
    assert result.loc[0, "iqr"] == pytest.approx(1.5)
    assert result.loc[0, "standard_deviation"] == pytest.approx(np.sqrt(1.25))
    assert result.loc[1, "perfectly_stable"]


def test_three_factor_higher_order_residual():
    design = pd.DataFrame(product((0, 1), repeat=3), columns=["A", "B", "C"])
    values = [(-1) ** sum(row) for row in design.itertuples(index=False, name=None)]
    result = factorial_attribution(pd.DataFrame([values]), design)
    assert result.contributions.loc[0, "interaction_residual"] == pytest.approx(1)
    assert result.contributions.iloc[0, :-1].sum() == pytest.approx(0)


def test_multiclass_experiment_and_attribution():
    X, y = load_iris(return_X_y=True)
    space = PipelineSpace(
        scaler=Choice(StandardScaler(), MinMaxScaler()),
        selector=ParamChoice(SelectKBest, score_func=[f_classif], k=[2, 4]),
        model=LogisticRegression(max_iter=1000, random_state=42),
    )
    experiment = space.fit(X, y)
    probabilities = experiment.predict_proba(X[:5])
    assert probabilities.shape == (5, 4, 3)
    report = experiment.audit(X[:5], class_index=2)
    assert report.dataframe.shape[0] == 5
    assert report.contributions.shape == (5, 4)
    assert report.contributions.sum(axis=1).between(0.999999, 1.000001).all()
    assert experiment.explain(0).metrics.majority_class in experiment.classes


def test_regression_experiment_and_exports(tmp_path):
    X, y = make_regression(n_samples=50, n_features=4, random_state=42)
    space = PipelineSpace(scale=Choice(StandardScaler(), MinMaxScaler()), model=Ridge())
    experiment = space.fit(X, y)
    report = experiment.audit(X[:3])
    assert experiment.predict(X[:3]).shape == (3, 2)
    assert (
        report.most_sensitive(1).index[0] == report.dataframe.prediction_range.idxmax()
    )
    assert report.sample(0).total_sum_of_squares >= 0
    report.to_csv(tmp_path / "audit.csv")
    report.to_json(tmp_path / "audit.json")
    assert len(pd.read_csv(tmp_path / "audit.csv")) == 3
    payload = json.loads((tmp_path / "audit.json").read_text())
    assert payload["metadata"]["task_type"] == "regression"
    assert len(payload["universes"]) == 2
    assert len(payload["attribution"]) == 3


def test_cache_equivalence_and_sampled_attribution_restriction():
    X, y = load_iris(return_X_y=True)
    space = PipelineSpace(
        first=Choice(StandardScaler(), MinMaxScaler()),
        second=Choice(StandardScaler(), MinMaxScaler()),
        model=LogisticRegression(max_iter=1000, random_state=42),
    )
    plain = space.fit(X, y, cache=False)
    cached = space.fit(X, y, cache=True)
    assert cached.metadata["cache_hits"] == 2
    np.testing.assert_allclose(
        plain.predict_proba(X[:10]), cached.predict_proba(X[:10])
    )
    pd.testing.assert_frame_equal(
        plain.audit(X[:10]).contributions, cached.audit(X[:10]).contributions
    )
    assert (
        plain.universes[0].pipeline["model"] is not plain.universes[1].pipeline["model"]
    )
    sampled = space.fit(X, y, strategy="random", n_universes=2, random_state=42)
    report = sampled.audit(X[:3])
    assert report.attribution is None
    with pytest.raises(ValueError, match="complete balanced"):
        report.sample(0)


def test_extended_demos_and_cli_info(capsys):
    wine = build_wine_demo()
    regression = build_regression_demo()
    assert wine.report.n_universes == 9
    assert wine.report.dataframe.shape[0] == 45
    assert regression.report.n_universes == 9
    assert regression.report.dataframe.shape[0] == 111
    assert main(["info"]) == 0
    assert "regression" in capsys.readouterr().out


def test_sampled_json_and_seeded_ids(tmp_path):
    X, y = load_iris(return_X_y=True)
    space = PipelineSpace(
        a=Choice(StandardScaler(), MinMaxScaler()),
        b=Choice(StandardScaler(), MinMaxScaler()),
        model=LogisticRegression(max_iter=1000),
    )
    first = space.fit(X, y, strategy="random", n_universes=2, random_state=4)
    second = space.fit(X, y, strategy="random", n_universes=2, random_state=4)
    assert first.metadata["universe_ids"] == second.metadata["universe_ids"]
    report = first.audit(X[:2])
    report.to_json(tmp_path / "sampled.json")
    assert json.loads((tmp_path / "sampled.json").read_text())["attribution"] is None

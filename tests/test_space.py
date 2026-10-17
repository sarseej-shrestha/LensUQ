from itertools import product

import pytest
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import LinearSVC

from lensuq import Choice, PipelineSpace


def make_space():
    return PipelineSpace(
        imputer=Choice(SimpleImputer(), SimpleImputer(strategy="median"), KNNImputer()),
        scaler=Choice(StandardScaler(), RobustScaler(), MinMaxScaler()),
        model=LogisticRegression(random_state=42),
    )


def test_full_product_and_order():
    space = make_space()
    assert space.n_universes == 9
    assert len(space.universes) == 9
    assert [tuple(u.choices.values()) for u in space.universes] == list(
        product(*space.factors.values())
    )
    assert space.metadata.equals(make_space().metadata)
    assert space.universes[0].is_nominal
    assert space.metadata.index.tolist() == list(range(9))


def test_no_shared_estimators():
    universes = make_space().universes
    for name in ("imputer", "scaler", "model"):
        assert len({id(u.pipeline[name]) for u in universes}) == 9


def test_fixed_chain_has_one_universe():
    space = PipelineSpace(scaler=StandardScaler(), model=LogisticRegression())
    assert space.n_universes == 1
    assert space.metadata.shape == (1, 0)
    assert space.universes[0].choices == {}


def test_space_snapshots_input():
    model = LogisticRegression(C=2)
    space = PipelineSpace(model=model)
    model.C = 10
    assert space.universes[0].pipeline["model"].C == 2


@pytest.mark.parametrize(
    "stages, error, message",
    [
        ({}, ValueError, "final stage"),
        (
            {"model": LogisticRegression(), "scale": StandardScaler()},
            ValueError,
            "final",
        ),
        (
            {"bad__name": StandardScaler(), "model": LogisticRegression()},
            ValueError,
            "name",
        ),
        (
            {"memory": StandardScaler(), "model": LogisticRegression()},
            ValueError,
            "name",
        ),
        ({"model": Choice(LogisticRegression())}, ValueError, "fixed"),
        ({"model": StandardScaler()}, TypeError, "classifier"),
        ({"model": LinearSVC()}, TypeError, "predict_proba"),
        (
            {"scale": "passthrough", "model": LogisticRegression()},
            TypeError,
            "transformers",
        ),
        (
            {"scale": Choice(LogisticRegression()), "model": LogisticRegression()},
            TypeError,
            "transformers",
        ),
    ],
)
def test_invalid_configurations_fail_early(stages, error, message):
    with pytest.raises(error, match=message):
        PipelineSpace(**stages)


def test_factor_named_universe_id_does_not_overwrite_metadata_index():
    space = PipelineSpace(
        universe_id=Choice(StandardScaler(), MinMaxScaler()),
        model=LogisticRegression(),
    )
    assert space.metadata.index.tolist() == [0, 1]
    assert space.metadata["universe_id"].tolist() == [
        "standard_scaler",
        "min_max_scaler",
    ]


def test_attribution_residual_name_is_reserved():
    with pytest.raises(ValueError, match="stage name"):
        PipelineSpace(
            interaction_residual=Choice(StandardScaler()), model=LogisticRegression()
        )

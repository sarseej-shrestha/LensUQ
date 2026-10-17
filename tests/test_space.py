from itertools import product

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

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

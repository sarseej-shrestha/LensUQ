import pytest
from sklearn.impute import KNNImputer, SimpleImputer

from lensuq.choice import Choice


def test_empty_choice_rejected():
    with pytest.raises(ValueError, match="at least one"):
        Choice()


def test_order_and_deterministic_labels():
    choice = Choice(
        SimpleImputer(strategy="mean"),
        SimpleImputer(strategy="median"),
        KNNImputer(),
        KNNImputer(),
    )
    assert choice.labels == (
        "simple_imputer_mean",
        "simple_imputer_median",
        "knn_imputer",
        "knn_imputer_2",
    )
    assert [option.strategy for option in choice.options[:2]] == ["mean", "median"]
    assert "simple_imputer_median" in repr(choice)


def test_templates_are_snapshots_and_never_shared():
    source = SimpleImputer()
    choice = Choice(source, name="Missing values", labels=["mean"])
    source.strategy = "median"
    option = choice.options[0]
    option.fit([[1], [2]])
    assert choice.options[0].strategy == "mean"
    assert not hasattr(choice.options[0], "statistics_")
    assert not hasattr(source, "statistics_")


@pytest.mark.parametrize("labels", [[], ["x", "x"], ["", "b"], "ab"])
def test_invalid_labels(labels):
    with pytest.raises(ValueError):
        Choice(SimpleImputer(), SimpleImputer(), labels=labels)


def test_non_estimator_rejected():
    with pytest.raises(TypeError, match="estimators"):
        Choice("mean")

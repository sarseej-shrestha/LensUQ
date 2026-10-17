import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from lensuq.universe import Universe


def test_steps_are_independently_cloned():
    scaler = StandardScaler().fit([[1], [2]])
    model = LogisticRegression()
    steps = [("scale", scaler), ("model", model)]
    first = Universe.from_steps(0, {"scale": "standard"}, steps)
    second = Universe.from_steps(1, {"scale": "standard"}, steps)
    assert first.is_nominal
    assert not second.is_nominal
    assert first.pipeline["scale"] is not second.pipeline["scale"]
    assert first.pipeline["model"] is not model
    assert not hasattr(first.pipeline["scale"], "mean_")
    assert hasattr(scaler, "mean_")
    with pytest.raises(TypeError):
        first.choices["scale"] = "robust"

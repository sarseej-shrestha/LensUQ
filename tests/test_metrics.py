import numpy as np
import pandas as pd
import pytest

from lensuq.metrics import prediction_metrics


def test_hand_computed_metrics():
    result = prediction_metrics([[0.2, 0.4, 0.6, 0.8]]).iloc[0]
    assert result.nominal_probability == 0.2
    assert result.mean_probability == pytest.approx(0.5)
    assert result.standard_deviation == pytest.approx(np.sqrt(0.05))
    assert result.minimum_probability == 0.2
    assert result.maximum_probability == 0.8
    assert result.probability_range == pytest.approx(0.6)
    assert result.positive_prediction_count == 2
    assert result.negative_prediction_count == 2
    assert result.positive_prediction_fraction == 0.5
    assert result.negative_prediction_fraction == 0.5
    assert result.decision_agreement == 0.5
    assert result.class_flip_count == 2
    assert result.decision_entropy == 1
    assert result.majority_decision == 1
    assert not result.perfectly_stable


def test_flips_are_relative_to_nominal_not_minority_count():
    row = prediction_metrics([[0.1, 0.7, 0.8, 0.9]]).iloc[0]
    assert row.class_flip_count == 3
    assert row.decision_agreement == 0.75
    assert row.decision_entropy == pytest.approx(0.8112781244591328)


@pytest.mark.parametrize("probabilities", [[[0.0]], [[1.0] * 9], [[0.1] * 9]])
def test_perfect_stability(probabilities):
    row = prediction_metrics(probabilities).iloc[0]
    assert row.standard_deviation == 0
    assert row.probability_range == 0
    assert row.class_flip_count == 0
    assert row.decision_entropy == 0
    assert row.decision_agreement == 1
    assert row.perfectly_stable


def test_threshold_and_index():
    values = pd.DataFrame([[0.3, 0.4, 0.5]], index=["sample"])
    result = prediction_metrics(values, threshold=0.4)
    assert result.index.tolist() == ["sample"]
    assert result.iloc[0].positive_prediction_count == 2


@pytest.mark.parametrize(
    "values", [[], [0.2], [[np.nan]], [[np.inf]], [[-0.1]], [[1.1]]]
)
def test_invalid_probabilities(values):
    with pytest.raises(ValueError):
        prediction_metrics(values)


@pytest.mark.parametrize("threshold", [-1, 2, np.nan, np.inf])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        prediction_metrics([[0.5]], threshold=threshold)


def test_complementary_vote_counts_have_exactly_equal_agreement():
    result = prediction_metrics([[0.1] * 6 + [0.9] * 3, [0.1] * 3 + [0.9] * 6])
    assert result.decision_agreement.iloc[0] == result.decision_agreement.iloc[1]
    assert result.decision_entropy.iloc[0] == result.decision_entropy.iloc[1]

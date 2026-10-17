from itertools import product

import numpy as np
import pandas as pd
import pytest

from lensuq.attribution import factorial_attribution


def decompose(values, coordinates, columns):
    design = pd.DataFrame(coordinates, columns=columns)
    probabilities = pd.DataFrame([values], index=["sample"])
    return factorial_attribution(probabilities, design)


def test_one_factor_hand_computation():
    result = decompose([0.2, 0.8], [["low"], ["high"]], ["imputer"])
    assert result.total_sum_of_squares.iloc[0] == pytest.approx(0.18)
    assert result.sums_of_squares.loc["sample", "imputer"] == pytest.approx(0.18)
    assert result.contributions.loc["sample", "imputer"] == pytest.approx(1)
    assert result.contributions.loc["sample", "interaction_residual"] == 0
    assert result.level_means.mean_probability.tolist() == [0.2, 0.8]


def test_two_factor_additive_hand_computation():
    # A changes probability by .4, B by .2: SS_A=.16, SS_B=.04.
    result = decompose([0.1, 0.3, 0.5, 0.7], product(["a", "b"], repeat=2), ["A", "B"])
    assert result.total_sum_of_squares.iloc[0] == pytest.approx(0.2)
    np.testing.assert_allclose(
        result.sums_of_squares.iloc[0], [0.16, 0.04, 0], atol=1e-15
    )
    np.testing.assert_allclose(result.contributions.iloc[0], [0.8, 0.2, 0], atol=1e-15)
    assert result.contributions.iloc[0].idxmax() == "A"


def test_pure_interaction():
    result = decompose([0.1, 0.9, 0.9, 0.1], product([0, 1], repeat=2), ["A", "B"])
    np.testing.assert_allclose(result.contributions.iloc[0], [0, 0, 1], atol=1e-15)


def test_mixed_main_effects_and_interaction():
    # Grand=.5, A=+/-.2, B=+/-.1, A:B=+/-.05. Total SS=.21.
    result = decompose([0.25, 0.35, 0.55, 0.85], product([0, 1], repeat=2), ["A", "B"])
    np.testing.assert_allclose(result.sums_of_squares.iloc[0], [0.16, 0.04, 0.01])
    np.testing.assert_allclose(result.contributions.iloc[0], np.array([16, 4, 1]) / 21)


def test_zero_variation_and_singleton_factor():
    result = decompose([0.1] * 9, product(range(3), repeat=2), ["A", "B"])
    assert result.perfectly_stable.iloc[0]
    assert result.total_sum_of_squares.iloc[0] == 0
    assert (result.contributions == 0).all().all()
    singleton = decompose([0.2, 0.8], [("fixed", 0), ("fixed", 1)], ["A", "B"])
    np.testing.assert_allclose(singleton.contributions.iloc[0], [0, 1, 0])


def test_no_factors():
    result = factorial_attribution(pd.DataFrame([[0.7]]), pd.DataFrame(index=[0]))
    assert result.perfectly_stable.iloc[0]
    assert result.contributions.iloc[0, 0] == 0
    assert result.level_means.empty


def test_random_three_factor_conservation():
    design = pd.DataFrame(
        product(range(2), range(3), range(4)), columns=["A", "B", "C"]
    )
    predictions = pd.DataFrame(np.random.default_rng(42).uniform(size=(20, 24)))
    result = factorial_attribution(predictions, design)
    np.testing.assert_allclose(result.contributions.sum(axis=1), 1, atol=1e-12)
    np.testing.assert_allclose(
        result.sums_of_squares.sum(axis=1), result.total_sum_of_squares
    )
    assert (result.contributions >= 0).all().all()


def test_small_nonzero_variation_is_not_discarded():
    result = decompose([0.5, 0.5 + 1e-12], [[0], [1]], ["A"])
    assert not result.perfectly_stable.iloc[0]
    assert result.contributions.iloc[0, 0] == pytest.approx(1)


@pytest.mark.parametrize(
    "rows", [[(0, 0), (0, 1), (1, 0)], [(0, 0), (0, 1), (1, 0), (1, 0)]]
)
def test_incomplete_or_duplicated_design_rejected(rows):
    with pytest.raises(ValueError, match="complete balanced"):
        decompose([0.5] * len(rows), rows, ["A", "B"])


def test_misaligned_universes_rejected():
    with pytest.raises(ValueError, match="match the design"):
        factorial_attribution(
            pd.DataFrame([[0.1, 0.9]]), pd.DataFrame({"A": [0, 1]}, index=[1, 0])
        )

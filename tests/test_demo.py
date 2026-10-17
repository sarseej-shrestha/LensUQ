from io import StringIO

import numpy as np
import pandas as pd
import pytest
from rich.console import Console
from sklearn.datasets import load_breast_cancer

from lensuq.demo import MISSING_FEATURES, introduce_missingness, run_demo


def test_missingness_is_deterministic_and_nonmutating():
    dataset = load_breast_cancer(as_frame=True)
    original = dataset.data.copy()
    corrupted = introduce_missingness(dataset.data)
    pd.testing.assert_frame_equal(corrupted, introduce_missingness(dataset.data))
    pd.testing.assert_frame_equal(dataset.data, original)
    assert not dataset.data.isna().any().any()
    assert corrupted.isna().sum().sum() == 532
    rate = corrupted.loc[:, list(MISSING_FEATURES)].isna().to_numpy().mean()
    assert 0.13 < rate < 0.17
    untouched = original.columns.difference(MISSING_FEATURES)
    pd.testing.assert_frame_equal(corrupted[untouched], original[untouched])


def test_full_breast_cancer_demonstration():
    output = StringIO()
    result = run_demo(Console(file=output, width=120, color_system=None))
    assert result.space.n_universes == len(result.experiment.universes) == 9
    assert result.training_observations == 455
    assert len(result.report.dataframe) == 114
    assert len(result.report.predictions) == 114 * 9
    ranked = result.experiment.most_sensitive()
    assert len(ranked) == 10
    sample = result.experiment.explain(ranked.index[0])
    assert len(sample.predictions) == 9
    assert sample.metrics.decision_agreement < 1
    assert sample.metrics.probability_range > 0
    np.testing.assert_allclose(result.report.contributions.sum(axis=1), 1, atol=1e-12)
    for text in (
        "not a medical finding",
        "Traditional view",
        "10 most",
        "All universe",
    ):
        assert text in output.getvalue()
    # All model settings are fixed across the nine pipelines.
    models = [u.pipeline["model"].get_params() for u in result.experiment.universes]
    assert all(params == models[0] for params in models)


@pytest.mark.parametrize("fraction", [-1, 2, np.nan])
def test_invalid_missingness_fraction(fraction):
    with pytest.raises(ValueError, match="fraction"):
        introduce_missingness(pd.DataFrame(), fraction=fraction)

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from lensuq import Choice, PipelineSpace


def test_factorial_experiment_end_to_end():
    values, y = make_classification(
        n_samples=120, n_features=6, n_informative=4, random_state=42
    )
    X = pd.DataFrame(values, index=[f"sample-{i}" for i in range(len(values))])
    X.iloc[::7, 0] = np.nan
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    space = PipelineSpace(
        imputer=Choice(SimpleImputer(), SimpleImputer(strategy="median"), KNNImputer()),
        scaler=Choice(StandardScaler(), RobustScaler(), MinMaxScaler()),
        model=LogisticRegression(max_iter=5000, random_state=42),
    )
    experiment = space.fit(X_train, y_train)
    report = experiment.audit(X_test)
    assert len(experiment.universes) == space.n_universes == 9
    assert len(report.dataframe) == len(X_test) == 24
    assert len(report.predictions) == 24 * 9
    np.testing.assert_allclose(report.contributions.sum(axis=1), 1, atol=1e-12)
    sample_index = experiment.most_sensitive(1).index[0]
    explanation = experiment.explain(sample_index)
    assert explanation.sample_index == sample_index
    assert len(explanation.predictions) == 9
    assert len(explanation.level_means) == 6
    pd.testing.assert_frame_equal(experiment.most_sensitive(), report.most_sensitive())
    # Refitting the space creates a new experiment with unchanged deterministic output.
    repeat = space.fit(X_train, y_train).audit(X_test)
    pd.testing.assert_frame_equal(report.dataframe, repeat.dataframe)

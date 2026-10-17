# LensUQ

**Measure how much a prediction depends on defensible preprocessing choices.**

LensUQ is a small Python library for pipeline-induced predictive variation.
Version 0.1 independently trains scikit-learn pipelines, audits each observation
across the resulting predictions, and attributes probability variation to
preprocessing factors.

## The problem

Many uncertainty analyses assume a fixed representation of the input. Yet mean,
median, and nearest-neighbor imputation can all be plausible choices, as can
standard, robust, and min-max scaling. Different choices can produce different
predictions for the same raw observation—even with the model family,
hyperparameters, and training partition held fixed.

For a user-defined set of defensible pipelines $\mathcal{P}$:

$$
p \in \mathcal{P}, \qquad \hat{y}_p = f_{\theta_p}(p(x)).
$$

Each $\theta_p$ is learned jointly with its own preprocessing pipeline, using
only training data. LensUQ examines the distribution induced by these choices.
The user defines which alternatives are defensible; the library does not infer
that scientific judgment or determine the correct pipeline.

| Approach | Question |
| --- | --- |
| Hyperparameter search | Which pipeline should I select? |
| LensUQ | How dependent is this prediction on the pipeline assumptions I could reasonably have selected? |

There is no scoring objective, best estimator, test-set selection, or leaderboard
of winning pipelines. The first declared combination is the **nominal universe**.
All universes receive equal weight, and each has an independently fitted model.

## Install and run

Python 3.11 or newer is required. From a checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python examples/breast_cancer_demo.py
# Equivalent packaged entry points:
lensuq demo
python -m lensuq demo
```

The demo uses scikit-learn's bundled breast cancer dataset and needs no data
download. Missingness is introduced artificially for the prototype to create a
controlled preprocessing-sensitivity experiment. **The demo is not a medical
finding.** Probabilities refer to the dataset's class **1 = benign**, not to
malignancy. Labels are unchanged.

## Use the library

```python
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from lensuq import Choice, PipelineSpace

# X_train/y_train and X_test come from a split performed before any fitting.
space = PipelineSpace(
    imputer=Choice(
        SimpleImputer(strategy="mean"),
        SimpleImputer(strategy="median"),
        KNNImputer(n_neighbors=5),
        labels=("mean", "median", "knn_5"),
    ),
    scaler=Choice(StandardScaler(), RobustScaler(), MinMaxScaler()),
    model=LogisticRegression(max_iter=5000, random_state=42),
)
assert space.n_universes == 9
experiment = space.fit(X_train, y_train)
report = experiment.audit(X_test, threshold=0.5)

report.render()
ranked = experiment.most_sensitive(n=10)  # pandas DataFrame
sample = experiment.explain(ranked.index[0])
sample.render()

summary = report.dataframe
disagreements = summary.loc[summary.decision_agreement < 1]
all_predictions = report.predictions     # samples × universes, long form
factor_fractions = report.contributions
factor_level_means = report.level_means
```

Keyword order defines the preprocessing chain; `model` must be last and fixed.
Explicit option labels are optional. Source estimators are cloned and never
fitted or mutated. `predict_proba(X)` returns a sample-by-universe DataFrame for
`classes_[1]`. Reports preserve unique, flat pandas indices; arrays receive a
positional index. `explain(index)` uses an **index label** from the latest audit.
Earlier report objects remain usable after another audit.

## What an audit measures

- Nominal and mean probability; population standard deviation; minimum, maximum,
  and range across universes.
- Positive/negative class counts and fractions at the chosen threshold.
- **Decision agreement:** the majority count divided by universe count.
  One means unanimity; one half means an even binary split.
- **Class flip count:** how many universes disagree with the nominal class.
- **Decision entropy:** Shannon entropy in bits of the class-vote fractions.
  It is not entropy of the average predicted probability.

Ranking uses agreement ascending, probability range descending, then standard
deviation descending. Exact ties retain input order. Decisions use `p >= threshold`;
an even vote selects the positive majority label as a documented tie convention.
Probability variation can exist even when every universe predicts the same class.

For each sample, the complete balanced factorial design permits orthogonal
main-effect sums of squares:

$$
SS_T = \sum_u (p_u-\bar p)^2, \qquad
SS_F = \sum_l n_l(\bar p_{F=l}-\bar p)^2.
$$

The factor contribution is $SS_F/SS_T$. The **interaction residual** is
$\max(0, SS_T-\sum_F SS_F)/SS_T$. It aggregates interactions of all orders and
does not identify specific interacting pairs. Contributions sum to approximately
one for nonconstant predictions; exact constant predictions receive zero
contributions and `perfectly_stable=True`.

## Actual prototype result

The fixed demonstration masks 532 of 3,414 eligible cells (15.58%) across six
features, then uses a stratified 455/114 split with seed 42. Of 114 held-out
observations, six have class disagreement across nine universes. The ranking
automatically selects original dataset row **385**:

```text
Nominal choices:        mean imputation + standard scaling
Nominal probability:   0.167908   (class 0)
Mean probability:      0.330423
Population std:        0.181219
Probability min/max:   0.148909 / 0.604103
Probability range:     0.455193
Decision agreement:    66.67%     (6 of 9 vote for class 0)
Class flips:           3
Decision entropy:      0.918296 bits

Attribution:
  imputer               0.0960%
  scaler               99.6223%
  interaction residual  0.2817%
```

These are sensitivity measurements within this selected space, not clinical
conclusions. Scaling changes the effective geometry of regularized logistic
regression even with fixed model settings. The attribution describes the whole
refitted pipeline response, not a causal effect of an isolated transformer.
Full values, universe predictions, and dependency versions are recorded in
[the result snapshot](docs/demo-results.json). Numerical output can vary slightly
between dependency versions and numerical backends.

## Architecture and limits

`Choice` → `PipelineSpace` → cloned `Universe` pipelines → fitted `Experiment`
→ `AuditReport` / `SampleReport`. Numerical metrics and factorial attribution
are independent modules. See [the API contract](docs/api.md) and
[the mathematical model](docs/methodology.md).

Version 0.1 supports binary classifiers with `predict_proba`, ordered tabular
preprocessing, discrete complete Cartesian products, and serial execution.
It is intended for small, inspectable spaces. It does not estimate calibrated
coverage, establish prediction correctness, separate training-data uncertainty,
or automatically identify defensible choices. Equal weights describe a declared
finite space, not posterior probabilities over pipelines. Duplicate choices
implicitly change that weighting. Seed all stochastic estimators when isolating
preprocessing variation.

The demonstration imputes before scaling, so KNN distances use original feature
units. Its synthetic missingness is not a clinical missing-data mechanism.
There is no multiclass/regression support, model comparison, optimization,
continuous space, DAG execution, distributed training, deep-learning backend,
web UI, or production deployment.

## Development and roadmap

```bash
python -m pytest
ruff check .
ruff format --check .
python examples/breast_cancer_demo.py
```

Tests cover estimator isolation, leakage boundaries, indexing, exact metric
examples, factorial conservation, interaction-only cases, and the complete
offline demo. CI exercises Python 3.11 and 3.14.

- **0.1:** complete binary sklearn prototype with reports and factorial attribution.
- **0.2:** richer factor metadata and explicit provenance; evaluate multiclass needs.
- **0.3:** investigate sampled spaces and decomposition under incomplete designs.
- **Research:** dependent choices, continuous factors, representation DAGs,
  scientific adapters, and joint data/pipeline/model uncertainty.

Open `vault/` as an Obsidian vault and start with
[00 Home](vault/00%20Home.md) for the project decisions and research agenda.

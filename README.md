# LensUQ

LensUQ measures how predictions vary across a user-defined set of preprocessing pipelines.

## The Problem

Preprocessing is often treated as fixed, although several representations may be reasonable. A model fitted after mean imputation can disagree with an independently fitted model using median imputation. That variation is easy to miss when only one pipeline is reported.

## The LensUQ Question

**How dependent is this prediction on preprocessing assumptions?**

For a declared pipeline space P, LensUQ fits an estimator separately in every selected universe and examines the predictions for the same held-out observations. The user decides which alternatives are defensible. LensUQ does not decide which choice is scientifically correct.

## What LensUQ Is Not

- **GridSearchCV:** there is no performance objective or winning configuration.
- **An ensemble:** universe predictions are not automatically combined for deployment.
- **An automatic preprocessing selector:** scientific suitability remains the user's judgment.

## Installation

Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Quick Start

```python
from lensuq import Choice, ParamChoice, PipelineSpace
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler

space = PipelineSpace(
    imputer=ParamChoice(SimpleImputer, strategy=["mean", "median"]),
    scaler=Choice(StandardScaler(), RobustScaler()),
    model=LogisticRegression(max_iter=1000, random_state=42),
)
# Split X and y before fitting. Transformers fit only on X_train.
experiment = space.fit(X_train, y_train, cache=True)
report = experiment.audit(X_test)
print(report.most_sensitive(5))
print(report.sample(report.most_sensitive(1).index[0]).contributions)
report.to_csv("samples.csv")
report.to_json("audit.json")
```

## Core Abstractions

`Choice` snapshots an ordered set of cloneable sklearn transformers. It supports explicit labels and `"passthrough"`. `ParamChoice` expands finite parameter lists in declaration order. `PipelineSpace` accepts any number of ordered preprocessing stages followed by a fixed `model`. A fixed stage is allowed; a single model-only space remains supported for compatibility. `Universe` holds an ID, factor labels, and its own cloned sklearn pipeline. `Experiment` owns fitted universes and execution metadata. `AuditReport` and `SampleReport` expose data independently of terminal rendering.

The first full-product combination is nominal. In a sampled run, the smallest selected Cartesian ID is nominal for that run. Neither is selected by test performance. Estimator family and settings remain fixed across universes.

## Full vs Sampled Spaces

`space.n_universes` gives the theoretical Cartesian size. Full enumeration uses rightmost-factor-fastest order. For a large space, sample unique IDs without constructing every pipeline:

```python
experiment = space.fit(
    X_train,
    y_train,
    strategy="random",
    n_universes=200,
    random_state=42,
)
```

The same seed and space give the same sorted IDs. Requests above the possible count fail. Sampled audits provide prediction metrics and exports; `report.attribution` is `None`, and `sample()` raises a clear error because classical orthogonal sums of squares are not valid for incomplete designs.

## Classification

Classifiers must implement `predict_proba`; probabilities are validated. Binary `experiment.predict_proba(X)` preserves the prototype sample-by-universe DataFrame for `classes_[1]`. The binary audit reports nominal and mean probability, population standard deviation, min, max, range, class counts and fractions, nominal flips, decision agreement, and vote entropy. At threshold `t`, equality predicts `classes_[1]`.

Multiclass `predict_proba(X)` returns a sample × universe × class NumPy array in `experiment.classes` order. The audit reports nominal and majority class, vote agreement and entropy, per-class frequencies, mean, population standard deviation, min, max, and range. Class decisions use maximum probability; tied votes select the first class in sklearn class order. Binary and multiclass sensitivity ranking sorts by lower decision agreement, then larger probability range, then larger standard deviation, then input order.

## Regression

Regressors require `predict`. `experiment.predict(X)` returns a sample-by-universe DataFrame. Audits report nominal, mean, population standard deviation, min, max, range, median, and IQR. Regression ranking sorts by larger raw prediction range, then larger standard deviation, then input order. Dispersion retains the target's units; no unstable near-zero normalization is used.

## Attribution

On a complete equally weighted Cartesian product, LensUQ decomposes the sum of squared prediction deviations from the grand mean. For each factor, the main-effect sum of squares is `sum_l n_l (mean_l - grand_mean)^2`. Each pair uses the two-factor cell mean after subtracting its two main effects. Remaining higher-order variation is reported as `interaction_residual`. Fractions sum to approximately one when total variation is positive; all fractions are zero for exactly constant predictions.

Binary attribution analyzes positive-class probability. Multiclass attribution analyzes the majority class probability for each observation by default, or a specified `class_index` passed to `audit`. Regression attribution analyzes the raw prediction. Contributions describe variation in the configured refitted pipeline experiment; they are not causal effects, p-values, or probabilities of error. Incomplete sampled designs have no attribution in this release.

## Execution Reuse

`cache=True` reuses fitted training transformations for identical preprocessing prefixes on the same training data. Each cached transformed array and fitted transformer is copied before reuse. Every universe still fits an independent model. The cache is local to one `fit` call and never crosses datasets. `cache=False` fits whole sklearn pipelines normally. Metadata records cache hits and training/analysis time.

## Exports

`report.to_dataframe()` returns sample metrics with the input index. `report.predictions`, `report.universes`, `report.samples`, and `report.attribution` expose other structured data. `to_csv(path)` writes sample metrics. `to_json(path)` writes metadata, universe coordinates, sample metrics, predictions, and attribution without fitted models. File I/O errors propagate to the caller.

## Demonstrations

All datasets are bundled with sklearn and use fixed seeds:

```bash
lensuq info
lensuq demo
lensuq demo breast-cancer
lensuq demo wine
lensuq demo regression
python examples/breast_cancer_demo.py
```

The breast cancer example uses artificial missingness, a 455/114 split, three imputers, three scalers, and logistic regression. **This is a controlled prototype demonstration of preprocessing sensitivity, not a medical finding.** The wine example uses three scalers and three feature counts with multiclass logistic regression. The diabetes example uses the same factor structure with Ridge regression. Each demo ranks actual observations; no result row is preselected.

## Testing

```bash
python -m pytest
ruff check .
ruff format --check .
```

CI runs these commands and the breast cancer demo on Python 3.11 and 3.14.

## Limitations

Spaces are finite and unconditional Cartesian products. Factor alternatives are equally weighted, so duplicate semantic options can change the implied measure. Attribution requires a complete balanced product. Prefix caching assumes sklearn-compatible transformers without external mutable state. Users must choose scientifically appropriate factors, train/test splits, and seeds for stochastic estimators. LensUQ does not estimate calibration, correctness, confidence intervals, or data-sampling uncertainty. Probability metrics require `predict_proba`.

## Future Work

Potential extensions include constrained spaces, attribution estimators for incomplete designs, repeated data and model seeds, richer provenance, and non-sklearn backends. These are research directions, not current features. The [research vault](vault/00%20Home.md) records the project rationale and open questions.

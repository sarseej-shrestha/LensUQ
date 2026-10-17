# Scope

## Implemented in 0.1

Python 3.11+, NumPy, pandas, scikit-learn, Rich, pytest, and Ruff. A source-layout package defines ordered transformer chains ending in one binary classifier with predict_proba. Choices are finite and exhaustive; every universe is independently fitted. Audits include indexed probability distributions, class-vote metrics, deterministic sample ranking, factorial main effects, an interaction residual, and terminal explanations.

The bundled offline demonstration uses nine imputation/scaling combinations and fixed logistic regression. Missingness is artificial and is not a medical finding. See [[Demo Design]].

## Later candidates

[[LensUQ 0.2]] considers factor rationale and provenance before widening computational scope. [[Roadmap]] postpones sampled spaces and dependent choices until their measurement assumptions can be stated and tested.

## Explicit non-goals

No automatic discovery of defensible choices, optimization objective, pipeline winner, model comparison, arbitrary DAG execution, continuous parameter search, distributed execution, PyTorch, TensorFlow, language-model pipeline, web interface, or production deployment. No clinical validation or uncertainty-coverage guarantee.

The public contract is dense tabular inputs, flat unique sample indices, and binary targets. General sparse containers, model persistence, multiclass and regression are not promised. A proposed extension must identify its scientific estimand and validation strategy before adding an abstraction. See [[Decisions]] and [[Long-Term Architecture]].

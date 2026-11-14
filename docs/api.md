# API contract

`Choice(*transformers, labels=None, name=None)` stores cloned ordered alternatives. `"passthrough"` is permitted. `ParamChoice(EstimatorClass, parameter=[values], ...)` expands a finite Cartesian product; labels show parameter names and values. Duplicate labels and empty levels are rejected.

`PipelineSpace(stage=Choice(...), ..., model=estimator)` uses keyword order for preprocessing order. `model` is fixed and must be a sklearn classifier with `predict_proba` or a regressor with `predict`. `n_universes`, `factors`, `metadata`, and `universes` expose the declared space. `select_universes(strategy="full"|"random", n_universes=None, random_state=None)` returns fresh universes. `fit(X, y, strategy=..., n_universes=..., random_state=..., cache=False)` fits independent models.

`Experiment.predict_proba(X)` returns a DataFrame for binary classification and a sample × universe × class array for multiclass. `predict(X)` returns a regression DataFrame. `audit(X, threshold=0.5, class_index=None)` returns an `AuditReport`. `most_sensitive(n)` and `explain(index)` use the latest successful audit.

`AuditReport.dataframe` and `to_dataframe()` contain sample metrics. `predictions`, `universes`, `samples`, `metadata`, and `attribution` expose further results. Binary long-form predictions include the positive-class `probability`; multiclass predictions include `winning_probability` and every `class_N_probability`. `contributions` and `level_means` require full factorial coverage. `most_sensitive(n)` ranks sample rows. `sample(index)` returns a `SampleReport` for full designs. `to_csv(path)` and `to_json(path)` write deterministic schema structures; timing fields are measured per run.

Sample indices must be unique, flat, and nonmissing. DataFrame X and Series y must have matching index order. The first executed universe is the nominal reference. Fitted universe pipelines are inspectable; mutating them invalidates experiment results.

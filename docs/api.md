# LensUQ 0.1 API contract

## Construction

`Choice(*estimators, name=None, labels=None)` snapshots unfitted clones, preserves
option order, and requires at least one sklearn estimator. `name` describes the
choice; the `PipelineSpace` keyword is the attribution factor name. Labels are
unique nonempty strings. Automatic labels use snake-case class names, optional
string `strategy`, and suffixes for collisions. Use explicit labels for durable
identifiers when changing a space. `options` returns fresh clones.

`PipelineSpace(**stages)` preserves keyword order and requires a final fixed
classifier named `model` with `predict_proba`. Earlier stages must be transformers,
either fixed or wrapped in `Choice`. The model's hyperparameters are held fixed.
The library cannot determine whether user-supplied alternatives are scientifically
defensible or semantically compatible; incompatible combinations fail at fit.

`space.n_universes` is the product of choice sizes, including singleton factors.
`space.metadata` contains one row per universe and factor-level labels.
`space.universes` creates fresh pipeline clones on each access. IDs start at zero;
the rightmost factor varies fastest. Universe zero is nominal. With no choices,
there is one universe and all probability variation is zero. `interaction_residual`
and sklearn Pipeline parameter names are reserved stage names.

`Universe` is a frozen metadata dataclass with `id`, read-only `choices`, and a
mutable sklearn `pipeline`. `Universe.from_steps` clones every estimator.
Fitted pipelines are exposed for inspection; refitting or mutating them invalidates
the experiment. Persistence of fitted experiment objects is outside the v0.1 API.

## Fitting and prediction

`space.fit(X_train, y_train)` returns an `Experiment`. It verifies binary targets,
consistent lengths, and matching row order for DataFrame/Series indices. Both
classes must be present. Every pipeline fits its own preprocessing and classifier
on its own copy of the same training data. Fit errors identify the failing universe.
No score or holdout labels are accepted or used for universe selection.

Inputs are dense numeric arrays or DataFrames supported by the selected sklearn
transformers. Sparse matrices, general object tables, and custom data containers
are not a supported contract in 0.1. Missing values require appropriate stages.
The caller must perform a suitable train/test split, keep schemas consistent,
and seed stochastic estimators. Cloning cannot repair a custom transformer's
external global state or a scientific protocol that leaks information upstream.

`experiment.predict_proba(X)` returns a DataFrame with shape
`(observations, universes)` and universe IDs as columns. It analyzes the second
sklearn probability column, identified by `experiment.positive_class` and
`experiment.classes`. The positive label need not be numeric. Predictions must
be finite, bounded probabilities with two columns summing to one.

DataFrame indices must be flat, unique, and nonmissing; array input receives a
`RangeIndex`. Empty audits are rejected. Prediction copies inputs for each
universe, protecting against in-place transformations. It does not refit any
pipeline or replace the most recent audit.

## Reports and selection

`experiment.audit(X, threshold=0.5)` returns an `AuditReport`. Thresholds are finite
values in `[0, 1]`; equality predicts the positive class. A failed audit leaves
the previous successful report intact. The experiment retains the latest report
for convenience; each returned report remains an independent snapshot.

| Operation | Result |
| --- | --- |
| `report.dataframe` | Per-sample descriptive metrics, decisions, class labels |
| `report.predictions` | Long table: `sample_index`, `universe_id`, `choice:<factor>`, `probability`, `predicted_class` |
| `report.contributions` | Factor fractions plus `interaction_residual` |
| `report.level_means` | Sample, factor, level, probability mean, universe count |
| `report.most_sensitive(n=10)` | Ranked DataFrame; zero returns an empty table |
| `report.sample(index)` | A `SampleReport`, resolved by index label |
| `report.render(console=None)` | Rich terminal summary |

Tabular report properties return copies. Use ordinary pandas sorting, `.loc`
filtering, and CSV export on those copies. `experiment.most_sensitive(n)` and
`experiment.explain(index)` delegate to the most recent audit and require one
to exist first. Ranking keys are agreement ascending, range descending, and
population standard deviation descending; final ties retain original audit order.

A `SampleReport` exposes `sample` (the raw feature row), `metrics`,
`nominal_choices`, all `predictions`, `contributions`, `sums_of_squares`,
`total_sum_of_squares`, `level_means`, `positive_class`, and `threshold`.
Its `render()` displays every universe, summary metrics, and attribution.

## Numerical functions

`lensuq.metrics.prediction_metrics(probabilities, threshold=0.5)` accepts a
nonempty sample-by-universe matrix. Column zero is nominal. DataFrame indices are
preserved. The returned decision codes are zero/one; experiments add the original
class labels. Standard deviation uses the finite-space population convention.

`lensuq.attribution.factorial_attribution(probabilities, design)` accepts a
probability DataFrame and a factor-coordinate DataFrame indexed by universe ID.
Probability columns must exactly equal the design index in the same order.
Incomplete, repeated, or missing factorial cells are rejected. The `Attribution`
result exposes contribution fractions, sums of squares, total sums of squares,
level means, and an exact-equality stability flag. See [methodology](methodology.md).

# Decisions

## ADR 001 — sklearn first

Accepted. Cloneable sklearn estimators and Pipelines define the current execution contract. PyTorch and other backends are deferred. [[Long-Term Architecture]] records possible adapters.

## ADR 002 — user-defined defensible spaces

Accepted. Domain experts choose levels; computational validity does not imply scientific validity. No automatic selector or performance winner is produced. See [[Vision]].

## ADR 003 — full and sampled execution

Accepted. Full products support exact orthogonal attribution. Random unique IDs reduce execution cost but provide metrics only. An incomplete-design attribution method requires separate research. See [[Execution Strategies]].

## ADR 004 — independent fitting

Accepted. Every universe clones preprocessing and fits its own model on the same training partition. In-place input mutation is isolated with copies. See [[Universe Model]].

## ADR 005 — classification and regression

Accepted. Classification requires `predict_proba` and supports binary and multiclass. Regression requires `predict`. Metrics preserve task-specific meaning. See [[Classification Analysis]] and [[Regression Analysis]].

## ADR 006 — main and pairwise decomposition

Accepted for complete Cartesian spaces. Report factor main effects, pairwise interactions, and a higher-order residual. Zero-variance outputs receive zero fractions. These are descriptive, not causal or inferential. See [[Attribution Model]].

## ADR 007 — safe prefix reuse

Accepted as an optional fit-local cache. Only identical ordered prefixes share transformed training results, and copies prevent fitted-object sharing. Cross-dataset or global caching is excluded. See [[Execution Graph and Caching]].

## ADR 008 — structured exports

Accepted. Reports expose DataFrames and write CSV/JSON without fitted models. JSON includes factor coordinates and execution metadata. See [[Result Objects]].

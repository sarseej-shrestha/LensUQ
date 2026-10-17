# Choice and Pipeline Space

A Choice stores an ordered tuple of cloned estimator templates and deterministic labels. Explicit labels are preferred for long-lived spaces; automatic labels combine class names, string strategies where available, and collision suffixes. A descriptive Choice name does not override its PipelineSpace keyword factor name.

PipelineSpace interprets keyword insertion order as execution order. The final stage is a fixed model with predict_proba. Earlier stages are transformers or Choices of transformers. Fixed stages are cloned in every universe but do not create attribution factors.

The Cartesian product enumerates levels with the rightmost factor varying fastest. Three imputers followed by three scalers produce IDs 0 through 8. ID 0 is the first option of every factor: the nominal universe. The metadata table contains universe IDs and factor-level coordinates without fitted estimators. Reading space.universes produces fresh pipelines; fitting the same space again creates a new experiment.

Empty choices, duplicate explicit labels, invalid step names, nontransformer preprocessing, a Choice model, and models lacking probabilities fail early. Semantic incompatibility between otherwise valid estimators can still surface at fit and is reported with universe context.

Complete coverage is necessary for the current [[Attribution Model]]. A one-level Choice contributes no main-effect variation; no Choices yield one fixed universe. Duplicating options is permitted but changes the equal-weight measure, so the scientific rationale must account for it. Dependent choices require a future design rather than silently dropping inconvenient cells.

Related: [[Terminology]], [[Universe Model]], [[Scope]].

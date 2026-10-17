# LensUQ 0.2

This is a candidate scope, not an implemented contract. The next release should improve scientific traceability before increasing combinatorial complexity.

Proposed additions are optional factor/level rationale, stable space identifiers, and audit provenance containing model parameters, dependency versions, positive-class meaning, threshold, and user-supplied split/data identifiers. Exports should be reconstructible without serializing fitted estimator internals or sensitive raw observations by default.

Multiclass support requires a design decision: analyze every class probability separately, focus on a declared class, or measure changes in a probability vector. Decision agreement and entropy would generalize, but per-class attribution alone may miss geometry across classes. A concrete use case and hand-computed tests should precede API commitments.

A compatible release should preserve ordered product semantics, nominal ID zero, independent fitting, binary metric definitions, and report snapshot behavior. Weighted or constrained spaces should not silently reuse the current factorial attribution.

Acceptance criteria: a documented metadata schema, round-trip export tests, a reproducible audit example, and migration notes for any changed semantics. [[Backlog]] supplies candidates; [[Research Questions]] keeps the scientific motivation explicit.

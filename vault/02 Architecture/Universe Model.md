# Universe Model

A universe has a Cartesian ID, immutable factor-choice mapping, and mutable sklearn Pipeline. Mutable fitted state belongs to that universe. `PipelineSpace` creates fresh clones; `Experiment` fits each model independently on the same training rows. Copying inputs protects against in-place transformers. A selected random subset preserves original Cartesian IDs, which may be nonconsecutive.

The first executed universe is the nominal reference. It is never chosen by test performance. An unseeded estimator can add stochastic variation outside the declared preprocessing factors; callers should set its seed. See [[Experiment Lifecycle]] and [[Testing Strategy]].

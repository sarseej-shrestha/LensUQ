# Problem Statement

A prediction is conditional on preprocessing decisions. When several representations are defensible, reporting only one can hide sensitivity to those decisions. LensUQ asks how the same held-out observation behaves across a user-defined finite pipeline space.

For each universe, the same training rows and model configuration are used, but the estimator is independently fitted on that universe's transformed representation. Equal universe weights describe the declared alternatives, not posterior probabilities or probabilities of correctness. The user remains responsible for scientific defensibility, valid splitting, and interpretation.

See [[Vision]], [[Scope]], [[Pipeline-Induced Uncertainty]], and [[Attribution Model]].

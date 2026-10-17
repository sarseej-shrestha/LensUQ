# Long-Term Architecture

The 0.1 chain is intentionally simple. Future architecture should preserve the invariant that a universe includes both representation construction and the model consistently trained on it.

## Choice DAG and reuse

A preprocessing choice DAG could represent conditional branches and shared transforms. Nodes would need explicit input/output contracts, fit scope, and provenance. Common-subexpression caching may reuse pure transformations only when their training inputs and fitted parameters are identical. Cache keys must include split, upstream state, parameters, and versions. Mutable fitted estimators must not become shared training state accidentally.

## Attribution and scale

Sobol or functional ANOVA approaches are candidates for continuous or large independent-factor spaces, after verifying their assumptions and literature. Correlated choices need a justified alternative decomposition. Sampling huge configuration spaces requires explicit target weights, reproducible sampling, coverage diagnostics, and approximation error. Compare with exhaustive small designs before replacing exact computation.

## Other scientific representations

Image and signal preprocessing may introduce spatial alignment, variable lengths, channel semantics, and sample exclusion. Adapters need stable sample identities and explicit failure policies. PyTorch support would need model factories, controlled initialization and training seeds, device handling, and consistent refitting under every representation. Scientific pipeline adapters should expose fit/transform boundaries rather than wrapping arbitrary scripts as opaque steps.

## Combined uncertainty

Combining data, pipeline, and model uncertainty requires separate axes for resamples, preprocessing choices, and model/seed configurations. A hierarchical or crossed design could expose conditional components and interactions. Flattening all runs into a single ensemble would hide which assumptions generated the variation.

These ideas remain proposals. [[Open Problems]] states the statistical obstacles, [[Future Research]] suggests studies, and [[Scope]] protects the working prototype from premature generalization.

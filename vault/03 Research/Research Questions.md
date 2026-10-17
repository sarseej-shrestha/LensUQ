# Research Questions

1. When does preprocessing sensitivity reveal practically important instability? Compare audit rankings with task-specific decision costs and externally reviewed failure cases, not only overall accuracy.
2. How should scientists define defensible pipeline spaces? Record domain rationale and whether combinations remain valid together; compare independently specified spaces rather than retrospectively selecting dramatic choices.
3. Can pipeline-induced variation be separated from model uncertainty? A crossed experiment with repeated seeds and data resamples could estimate components, but dependencies between representation and fitting need explicit treatment.
4. How should correlated or non-factorial choices be handled? Constraints change the reference measure and destroy the simple orthogonal decomposition in [[Attribution Model]].
5. How can continuous preprocessing choices be represented? Discretization needs convergence checks and a justified measure; arbitrary grids can overweight regions.
6. How should uncertainty across model and preprocessing spaces interact? Averaging them together conceals the source of disagreement. A hierarchical or crossed analysis might preserve attribution.
7. Does low sensitivity predict reliability under distribution shift? This is an empirical hypothesis; shared misspecification can produce stable errors.
8. How should humans act on a sensitive prediction? A useful explanation should expose alternatives and their effects without implying that disagreement alone determines an intervention.

[[Future Research]] organizes studies; [[Open Problems]] lists obstacles. The prototype tests numerical correctness, not these broader scientific hypotheses.

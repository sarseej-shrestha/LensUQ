# Scope

## Implemented in 0.2

Python 3.11+ sklearn-first library with finite ordered `Choice` and `ParamChoice` factors, full and reproducible random universe selection, independent estimator fitting, binary and multiclass probability audits, regression audits, sensitivity ranking, full-factorial main and pairwise attribution, higher-order residual, optional training-prefix cache, DataFrame/CSV/JSON reports, three offline demos, CLI, and CI.

## Boundaries

The user defines scientifically defensible alternatives and a valid split. LensUQ measures variation over those choices; it does not select a correct pipeline, calculate error probability, calibrate uncertainty, or combine predictions into a deployment estimator. Sampled designs do not receive classical factorial attribution. Constrained spaces, uneven weights, continuous priors, repeated seeds, non-sklearn backends, and scientific adapters remain research work. See [[Decisions]], [[Open Problems]], and [[Long-Term Architecture]].

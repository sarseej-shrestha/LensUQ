# Experiment Lifecycle

1. Define defensible preprocessing alternatives and a fixed compatible estimator.
2. Split data before fitting learned preprocessing.
3. Fit full or sampled universes on the same training rows.
4. Predict each held-out observation with every fitted pipeline.
5. Audit classification or regression metrics.
6. For complete products, decompose main, pairwise, and higher-order residual variation.
7. Rank observations and export results.

`Experiment` stores execution metadata, fitted universes, and the latest successful audit. Earlier reports remain usable. Sample indices remain aligned. See [[Result Objects]] and [[Attribution Model]].

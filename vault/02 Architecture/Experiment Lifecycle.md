# Experiment Lifecycle

1. Define the raw observations, labels, feature schema, and scientific rationale for alternatives.
2. Split observations before fitting any data-dependent transformation.
3. Declare the ordered space and fixed seeded model.
4. Build independent universe pipelines and fit each on the same training rows.
5. Predict the same held-out raw observations in every fitted universe.
6. Compute probability summaries, class-vote metrics, and factorial attribution.
7. Rank observations for inspection and explain individual cases.

In the demonstration only, artificial missingness is injected before the split. Its mask uses a fixed random generator and no learned statistics or labels. Both partitions are therefore drawn from the same deliberately corrupted dataset. This exception concerns a synthetic data-generation step, not preprocessing fitted globally. See [[Demo Design]].

Experiment.predict_proba returns a sample-by-universe matrix. Audit adds long-form coordinates, thresholded original class labels, and per-sample analysis. Unique flat DataFrame indices persist; arrays receive positions. The positive class is the second sklearn class and the threshold convention includes equality.

The latest successful report supports experiment.explain and experiment.most_sensitive. An unsuccessful audit does not replace it, and earlier returned reports remain available. No test labels are needed for audit, so no test performance can select the nominal universe.

Leakage checks and reproducibility evidence belong in [[Validation Strategy]]. Train/test grouping or temporal structure remains the caller's responsibility; the library does not design the scientific split.

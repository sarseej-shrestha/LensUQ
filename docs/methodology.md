# Measurement methodology

LensUQ measures variation over a finite user-defined set of preprocessing alternatives. Each universe receives the same training rows and an independently cloned estimator. Each preprocessing chain is fitted only on training data. Test observations are transformed with that universe's fitted chain. The framework does not rank pipelines by predictive performance.

For a sample, classification decisions are votes across universes. Decision agreement is the largest class count divided by the universe count. Decision entropy is Shannon entropy in bits of class frequencies. Probability dispersion uses population standard deviation. Multiclass probability metrics are computed for every class separately. Regression dispersion uses raw target units and percentile IQR.

On a complete Cartesian product, all factor contrasts are orthogonal under equal universe weighting. For values v_u, total SS is sum `(v_u - mean(v))²`. A factor main effect is sum over levels of `n_l (mean_l - mean(v))²`. For pair A,B, each cell contributes `n_ab (mean_ab - mean_a - mean_b + mean(v))²`. The residual is the remaining total SS after main and pairwise effects and may contain third and higher order interactions. Numerical roundoff is clipped at zero. Exactly constant values have zero fractions.

For multiclass, default attribution selects each observation's majority class probability; `class_index` selects a single probability column across all observations. For binary, the second class probability is used. For regression, the predicted value is used. Sampled designs retain their sample metrics but do not receive this decomposition. No inferential, causal, or calibration claim follows from these fractions.

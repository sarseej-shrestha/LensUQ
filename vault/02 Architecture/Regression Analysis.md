# Regression Analysis

Regressors require `predict`. Each held-out row receives one prediction per universe. Metrics are nominal, mean, population standard deviation, min, max, range, median, and percentile IQR. Ranking prioritizes larger raw range, then larger standard deviation. Units remain the target's units; normalization around zero is avoided.

Full-factorial attribution decomposes raw prediction variation. It does not measure prediction correctness. See [[Attribution Model]] and [[Result Objects]].

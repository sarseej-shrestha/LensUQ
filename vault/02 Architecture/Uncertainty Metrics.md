# Uncertainty Metrics

This historical note introduced binary probability and vote metrics in the prototype. Current task-specific definitions are in [[Classification Analysis]] and [[Regression Analysis]]. The common principle is to summarize a finite declared prediction distribution without claiming calibrated uncertainty or correctness.

Population standard deviation uses `ddof=0`. Classification decision agreement uses the largest integer vote count divided once by the universe count, preserving symmetry. Decision entropy is Shannon entropy of class frequencies in bits, not entropy of the mean probability. Exactly identical predictions are perfectly stable even when vote agreement alone would be less informative.

Ranking rules are recorded in [[Result Objects]]. Factor contributions are defined in [[Attribution Model]].

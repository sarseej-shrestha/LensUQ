# Pipeline-Induced Uncertainty

Treating preprocessing configuration as an uncertainty space makes a hidden conditioning assumption explicit. The output is conditional on a declared collection of decisions and a rule assigning them weight. In 0.1 that rule is uniform mass over a finite product.

A single observation can have a broad probability range even when every pipeline makes the same class decision. Conversely, a narrow range centered on a threshold can produce class disagreement. Both views matter and motivate the separate summaries in [[Uncertainty Metrics]].

The empirical distribution is epistemic in an informal sense—it reflects unresolved workflow choices—but it is not automatically a Bayesian posterior, a calibrated predictive distribution, or an estimate of aleatoric variability. Scientific meaning depends on why alternatives belong in the space. Repeating an option changes its weight without adding knowledge, illustrating why space design is part of the estimand.

Representation and training cannot be separated mechanically: different transforms change fitted parameters even under the same model settings. LensUQ's unit is therefore the entire consistently fitted universe. Its attribution answers how probability responds to choice factors over that refitting procedure.

A longer-term framework could distinguish variation from raw-data sampling, pipeline assumptions, training randomness, and model specification. It should expose conditional components and interactions rather than collapse everything into one number. [[Research Questions]] and [[Long-Term Architecture]] describe possible designs; no such decomposition is claimed by 0.1.

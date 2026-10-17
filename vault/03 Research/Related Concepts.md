# Related Concepts

These are conceptual relationships, not claims of identity or novelty. Literature details have not been verified for this prototype; the tasks below must be completed before adding academic citations.

| Concept | Connection | Important distinction |
| --- | --- | --- |
| Multiverse analysis | Studies conclusions across reasonable analytical choices. | LensUQ's prototype unit is an individual predictive probability from an independently fitted pipeline. |
| Specification curves | Make variation across specifications visible. | LensUQ adds a balanced-factorial probability decomposition; it does not implement inferential specification-curve tests. |
| Uncertainty quantification | Asks what is unresolved about a prediction. | Finite-space disagreement alone supplies no calibrated coverage. |
| Sensitivity analysis | Measures response to changing inputs or assumptions. | The factors here are preprocessing implementations and include model refitting. |
| Hyperparameter optimization | Evaluates multiple pipeline configurations. | Its selection objective differs from auditing dependence on alternatives without a winner. |
| Model ensembles | Collect multiple predictions, often to improve an aggregate. | LensUQ holds the model configuration fixed and treats the distribution as evidence to inspect, not merely average away. |
| Robustness analysis | Examines stability under perturbations. | Preprocessing choices are workflow assumptions, not necessarily local perturbations or adversarial attacks. |

## Literature verification queue

- TODO: verify foundational multiverse-analysis and specification-curve papers, their assumptions, and terminology against primary sources.
- TODO: verify functional ANOVA and Sobol decomposition references, especially dependence and weighting assumptions.
- TODO: locate predictive-pipeline sensitivity studies and assess overlap before making novelty claims.
- TODO: distinguish repeated-seed ensemble variability from preprocessing-induced effects in empirical comparisons.

[[Pipeline-Induced Uncertainty]] defines the intended concept. [[Future Research]] turns these connections into studies and [[Attribution Model]] states what the current estimator actually computes.

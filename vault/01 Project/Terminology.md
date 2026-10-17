# Terminology

| Term | Meaning in LensUQ |
| --- | --- |
| Pipeline universe | One preprocessing combination and the classifier independently fitted with it. |
| Pipeline space | The user-defined collection of alternatives; a complete discrete product in 0.1. |
| Factor / level | A Choice stage / one labeled option in that stage. Fixed stages are not factors. |
| Preprocessing sensitivity | Dependence of a prediction on the specified preprocessing alternatives. |
| Pipeline-induced predictive variation | The observed distribution of predictions over that finite space, conditional on data and training policy. |
| Nominal universe | The first declared combination, ID zero; a reference, not a selected winner. |
| Decision agreement | Majority class count divided by universe count at the declared threshold. |
| Class flip count | Number of universes whose thresholded decision differs from the nominal decision. |
| Factor attribution | A factor's main-effect sum of squares divided by total probability sum of squares. |
| Interaction residual | Variation left after all main effects; aggregates interactions of every order. |
| Perfectly stable | Exactly equal probabilities across the audited finite space; not a global robustness claim. |

The positive class means sklearn classes_[1], not necessarily a clinically adverse outcome. The demo's positive class is benign. A probability is not automatically a calibrated measure of correctness.

[[Uncertainty Metrics]] supplies equations. [[Attribution Model]] states when decomposition is valid. [[Related Concepts]] distinguishes neighboring research ideas.

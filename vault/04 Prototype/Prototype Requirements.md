# Prototype Requirements

The 0.1 contract is a complete, inspectable sklearn implementation, not a configuration optimizer.

- Choice preserves order, rejects emptiness, supplies deterministic labels, and snapshots cloned estimators.
- PipelineSpace validates an ordered transformer chain with a fixed probabilistic binary classifier and generates the complete product. Universe zero is nominal.
- Every universe fits its own preprocessing and classifier on the same training partition. No global preprocessing, test leakage, or reused fitted classifier is permitted.
- Experiment predicts every observation in every universe, retains class meanings and sample indices, and returns structured audits.
- Audits provide probability extent, population standard deviation, class fractions, agreement, flips relative to nominal, and decision entropy.
- Ranking is deterministic and documented; explanations expose all universe predictions, factor contributions, and level means.
- Attribution uses balanced-factorial main effects plus an aggregate interaction residual, with explicit zero-variance behavior.
- The offline breast cancer demonstration uses three imputers, three scalers, fixed logistic regression, artificial missingness, and one seeded stratified split.
- Tests, lint, formatting, editable installation, terminal entry points, README, and a navigable project vault accompany the code.

No training score selects a universe. No medical conclusion is inferred. The precise exclusions are in [[Scope]]. [[Validation Strategy]] and [[Testing Strategy]] define acceptance evidence; [[Prototype Results]] records it.

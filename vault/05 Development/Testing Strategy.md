# Testing Strategy

Tests should expose violations of scientific invariants, not just mirror implementation branches. Small analytic examples are the numerical oracle; the full demo verifies that those components compose.

Choice and space tests cover empty options, ordered stable labels, explicit-label validation, product coordinates, source snapshots, independent clones, fixed stages, and early invalid-configuration errors. Universe tests include templates already fitted before construction.

Experiment tests train toy binary data, preserve nonconsecutive labels, reject ambiguous indices, and compare probability columns directly with fitted pipelines. Input copies are tested using in-place transformers. Training statistics remain unchanged by held-out predictions; original estimators remain unfitted. Errors carry universe context.

Metric tests distinguish nominal flips from minority counts, entropy of votes from probability entropy, and probability constancy from decision unanimity. Complementary class counts must produce exactly equal agreement. Boundary thresholds and invalid probabilities receive explicit cases.

Attribution tests cover one factor, additive two-factor responses, a dominant factor, pure and mixed interactions, singleton factors, no factors, exact zero variation, tiny nonzero variation, and conservation on seeded larger products. Missing or duplicated combinations and misaligned IDs are rejected.

Report tests verify independent snapshots, class-label mapping, deterministic ties, lookup by label, rich rendering, and failed-audit state retention. The integration suite includes the full nine-universe artificial-missingness demo without hardcoding a dramatic row.

Run pytest with warnings as errors, then Ruff lint and formatting. CI checks the minimum supported Python and a current Python. [[Validation Strategy]] adds packaging and result-provenance checks; [[Prototype Results]] records actual outcomes.

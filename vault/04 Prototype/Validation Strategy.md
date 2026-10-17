# Validation Strategy

Correctness has three distinct layers: configuration independence, numerical validity, and end-to-end reproducibility. Passing one layer does not substitute for another.

Configuration tests enumerate the full product and check exact ordering, independent estimator identities, unfitted source templates, and reuse of one training partition. Adversarial copy=False transformers verify that input mutation cannot carry state between universes. Held-out prediction must not update fitted statistics.

Numerical tests use hand-computed metric and attribution examples, including pure interactions and exactly constant probabilities. Random three-factor grids check conservation of sums of squares. Tiny nonzero differences must not be silently treated as zero. Invalid probabilities, misaligned universe IDs, and incomplete products must fail clearly.

Integration tests train and audit both synthetic binary data and the full nine-universe offline demonstration. They verify observation counts, indexed outputs, ranking, explanation, model settings, and output rendering. The demonstration selects its sample algorithmically, so dependency updates need not preserve one hardcoded row in tests.

Release checks run pytest with warnings treated as errors, Ruff lint and format checks, the example script, editable installation, and the packaged CLI. CI covers Python 3.11 and 3.14. Build artifacts and caches are excluded from version control. Wiki links and actual result snapshots are checked for consistency.

These checks establish implementation behavior. Scientific validity of a proposed space, calibration, clinical relevance, and novelty require separate studies described in [[Research Questions]] and [[Future Research]].

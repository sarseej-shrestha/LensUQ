# Prototype Results

Measured local run of LensUQ 0.1.0 on Python 3.14.3, NumPy 2.5.4, pandas 3.0.6, and scikit-learn 1.9.1. The machine-readable snapshot is `docs/demo-results.json` at the repository root. Dependency and numerical-backend differences can shift the final digits.

Missingness is artificial and the demo is not a medical finding. Probability refers to class 1 (benign). See [[Demo Design]] for the exact mask and training protocol.

## Observed audit

- Training / test observations: 455 / 114.
- Universes: 9; every one fits its own classifier.
- Masked cells: 532 / 3414 eligible cells.
- Test observations with class disagreement: 6.
- Automatically selected most sensitive original row: 385.
- Nominal: mean imputation plus standard scaling, probability 0.167908010, class 0.
- Mean / population std: 0.330422618 / 0.181218645.
- Minimum / maximum: 0.148909460 / 0.604102579.
- Range: 0.455193119.
- Agreement: 66.666667%; 6 negative and 3 positive votes.
- Class flips relative to nominal: 3.
- Decision entropy: 0.918295834 bits.

## Attribution

| Component | Contribution |
| --- | ---: |
| Imputer | 0.095999% |
| Scaler | 99.622265% |
| Interaction residual | 0.281735% |

Total probability sum of squares: 0.295561775723. This is a descriptive partition of the selected finite space. Scaling dominates this observation; it does not follow that imputation is unimportant for every observation or every dataset.

## Verification

The final local suite passed **94 tests** with no warnings on Python 3.14.3. Ruff lint and formatting passed. Editable installation, the standalone example, and the packaged CLI ran successfully. A source archive and platform-independent wheel were built successfully; build artifacts remain outside version control. All links across the 26 vault notes resolve.

Reproduce from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
ruff check .
ruff format --check .
python examples/breast_cancer_demo.py
lensuq demo
```

[[Validation Strategy]] explains what these checks establish. [[Open Problems]] records what the prototype does not establish.

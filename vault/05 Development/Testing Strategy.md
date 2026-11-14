# Testing Strategy

Mathematical tests use hand-computable binary, multiclass, and regression metrics, additive main effects, pure pair interactions, pure three-factor residual, and zero variance. Space tests cover ordered factors, finite parameter expansion, unique reproducible sampling, and invalid counts. Execution tests inspect independent model state and compare cached with uncached predictions and attribution. Report tests verify index preservation and CSV/JSON structures. CLI and demos run offline.

Run `python -m pytest`, `ruff check .`, and `ruff format --check .`. CI uses Python 3.11 and 3.14. A fresh editable install is part of release validation. See [[Validation Strategy]] and [[Prototype Results]] for the historical baseline.

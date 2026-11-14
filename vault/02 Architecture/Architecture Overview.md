# Architecture Overview

Define: [[Choice and Pipeline Space]] stores ordered cloneable alternatives. Explore: [[Execution Strategies]] selects full or reproducibly sampled Cartesian IDs. Execute: [[Universe Model]] fits each estimator independently; [[Execution Graph and Caching]] can reuse safe fitted training prefixes. Analyze: [[Classification Analysis]], [[Regression Analysis]], and [[Attribution Model]] produce metrics. Report: [[Result Objects]] exports DataFrames, CSV, and JSON.

The code follows these boundaries in `choice.py`, `space.py`, `universe.py`, `experiment.py`, `metrics.py`, `attribution.py`, and `report.py`. `demo.py`, `extended_demos.py`, and `cli.py` provide offline examples. See [[Experiment Lifecycle]] for data flow and leakage boundaries.

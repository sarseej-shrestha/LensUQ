# Choice and Pipeline Space

`Choice` snapshots ordered sklearn transformer templates with labels. `ParamChoice` expands finite parameter lists. `"passthrough"` can be a level. Keyword order in `PipelineSpace` defines preprocessing order; `model` is the final fixed estimator. Arbitrary factor counts are supported. Fixed preprocessing stages do not become attribution factors.

The theoretical size is the product of factor level counts. Full IDs follow rightmost-factor-fastest order. Random exploration samples unique IDs with a local seeded generator. The model-only baseline remains available for compatibility. Each universe receives cloned estimators; see [[Universe Model]] and [[Execution Strategies]].

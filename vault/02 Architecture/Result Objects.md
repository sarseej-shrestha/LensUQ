# Result Objects

`AuditReport` exposes sample metrics (`dataframe`/`to_dataframe`), long-form predictions, raw samples, universe coordinates, metadata, and attribution. `SampleReport` exposes one observation's metrics, predictions, and full-factorial decomposition. CSV contains sample metrics; JSON contains metadata, universes, metrics, predictions, and attribution without fitted models. Execution times are measured values and vary by run.

Sample ranking is deterministic: classification uses agreement ascending, maximum probability range descending, standard deviation descending; regression uses raw range descending, standard deviation descending. Exact ties retain audit order. See [[Classification Analysis]] and [[Regression Analysis]].

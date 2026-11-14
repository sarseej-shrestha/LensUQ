# Long-Term Architecture

Future versions may support constrained spaces, weighted estimands, repeated data and model seeds, richer provenance, and additional estimator backends. Each extension must preserve the core invariant: a universe owns both its preprocessing representation and the estimator trained on that representation. Sampled and constrained attribution need a stated reference measure and numerical validation.

The current implementation includes a small safe prefix cache, not a general representation DAG. PyTorch integration is deferred. See [[Execution Graph and Caching]], [[Open Problems]], and [[Decisions]].

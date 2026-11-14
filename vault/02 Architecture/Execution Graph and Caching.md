# Execution Graph and Caching

`cache=True` stores fitted training transformations keyed by the ordered prefix of stage names and factor labels within one fit call. Reused fitted transformers and transformed arrays are deep-copied into each universe. Models always fit independently. The cache never spans datasets or fit calls. `cache=False` uses ordinary full sklearn Pipeline fitting.

The cache is a small prefix reuse layer, not a general execution DAG. It assumes sklearn-compatible transformers without external mutable state. Equivalence tests compare predictions, metrics, and attribution with caching off and on. Metadata records cache hits. See [[Universe Model]] and [[Testing Strategy]].

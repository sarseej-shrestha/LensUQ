# Execution Strategies

`strategy="full"` enumerates every Cartesian ID. `strategy="random"` draws a specified number of unique IDs using `random_state`; sampled IDs are sorted for stable execution. Mixed-radix decoding constructs only selected pipelines, so selection does not require full enumeration. The requested sample count must fit the space.

Sampled audits provide metrics and exports. Classical orthogonal attribution is withheld for sampled or unbalanced designs. A future incomplete-design estimator requires an explicit reference measure and validation; see [[Open Problems]] and [[Attribution Model]].

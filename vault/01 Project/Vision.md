# Vision

LensUQ should make preprocessing assumptions visible at the level of individual predictions. A scientist should be able to ask whether a conclusion survives plausible changes to how raw observations become model inputs, and inspect which decisions explain the disagreement.

The motivating problem is broader than missing-value handling. Signal filtering, image normalization, quality exclusion, feature extraction, and aggregation can each encode consequential assumptions. A prediction conditional on one representation can obscure this dependence.

Model uncertainty concerns the model or its learned parameters conditional on a representation and training procedure. Pipeline uncertainty concerns the representation-producing decisions themselves. These sources can interact: a transformation changes the fitting problem and therefore the learned parameters. LensUQ does not claim a clean causal separation merely by holding model settings fixed.

The long-term goal is an inspectable scientific tool with explicit provenance, meaningful space definitions, and honest limits. The current implementation is intentionally narrower: [[Scope]] and [[Prototype Requirements]]. Its value is demonstrating the question with a correct lifecycle and transparent [[Attribution Model]], not claiming to solve all uncertainty quantification.

Success means a user can explain which assumptions matter, why those assumptions were included, and what remains unknown. See [[Research Questions]] for how that goal becomes an evaluation agenda.

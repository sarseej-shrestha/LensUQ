# Open Problems

## Defensible alternatives and reference measures

Which alternatives deserve inclusion, and how much weight should they receive? A full Cartesian product can contain combinations that no practitioner would defend. Removing those cells makes the prototype attribution invalid. Domain rationale, constraints, and the induced reference measure must be represented together.

## Attribution outside a balanced product

Dependent factors and uneven weights make main effects nonorthogonal. Several decomposition conventions may allocate shared variation differently. A future API must identify its convention and explain sensitivity to it. Dropping failed universes silently is unacceptable.

## Multiple sources of variation

Seeds, training resamples, model specifications, and preprocessing can interact. Identical random seeds do not guarantee identical stochastic paths when transformed data differ. Crossing these dimensions increases cost rapidly and requires a clear replication unit.

## Meaningful action

What magnitude matters depends on the downstream decision and threshold. A high relative attribution can arise from negligible absolute variation. Agreement does not imply correctness, and no universal instability cutoff has been justified.

## Scale and provenance

Large spaces need approximate computation with error diagnostics. Cached transforms must preserve fit boundaries and avoid shared mutable state. Scientific audits need provenance sufficient to reconstruct both space rationale and numerical results.

See [[Research Questions]], [[Backlog]], and [[Long-Term Architecture]]. These are research and design issues rather than defects that can be resolved by adding a single metric.

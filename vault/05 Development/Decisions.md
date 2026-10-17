# Decisions

## ADR 001 — sklearn first

Accepted for 0.1. sklearn provides cloneable estimators, ordered Pipelines, and a compact binary-probability contract. Consequence: no deep-learning adapters, graph engine, or arbitrary containers in the first release.

## ADR 002 — discrete full factorial first

Accepted. Complete products are easy to inspect and make main effects orthogonal under equal weighting. Consequence: exponential cost, no dependent combinations, and no silent dropping of failed cells. [[Attribution Model]] is valid only under this contract.

## ADR 003 — fixed model family and settings

Accepted. The intended source of variation is preprocessing. Every representation gets an independently fitted copy of the same classifier configuration. Consequence: model search and performance selection are excluded, while interactions with fitting remain part of the measured response.

## ADR 004 — analyze probabilities before decisions

Accepted. Probability distributions retain information hidden by thresholding. Decisions add operational context; the two views remain separate. Consequence: the positive label and threshold are explicit, and the API avoids treating probabilities as correctness guarantees.

## ADR 005 — descriptive ANOVA partition

Accepted as a prototype estimator. Report main-effect fractions, absolute sums of squares, and an aggregate residual. Consequence: no p-values, no pair-specific claims, and no application to incomplete or weighted designs.

## ADR 006 — human-defined defensibility

Accepted. Scientific rationale cannot be inferred from estimator availability. Consequence: the library validates computation, not scientific suitability. [[LensUQ 0.2]] proposes rationale metadata.

## ADR 007 — copy inputs at universe boundaries

Accepted after considering copy=False transformers. Estimator cloning alone does not protect shared arrays. Consequence: extra memory and copying cost, traded for simple, testable isolation in small spaces.

## ADR 008 — rank by integer-derived agreement

Accepted. Complementary binary vote patterns must tie exactly before applying range and standard deviation. Consequence: agreement is calculated from the larger integer vote count divided once, avoiding subtraction roundoff that could reorder samples.

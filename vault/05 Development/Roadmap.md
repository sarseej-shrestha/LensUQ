# Roadmap

## 0.1 — inspectable prototype

Complete: discrete full-factorial sklearn spaces, isolated training, binary probability analysis, class-vote metrics, deterministic sample ranking, descriptive attribution, reports, terminal demonstration, tests, and project documentation. [[Prototype Results]] contains measured evidence.

## 0.2 — richer factors and provenance

Prioritize rationale, explicit space identities, reproducible audit metadata, and export conventions. Assess multiclass requirements without weakening the binary contract. Do not introduce dependent-factor attribution until its reference measure and estimator are justified. See [[LensUQ 0.2]].

## 0.3 — efficient spaces

Investigate stratified or low-discrepancy sampling of large products, approximation diagnostics, and safe reuse of shared computations. Validate against exhaustive small spaces and preserve leakage boundaries. Approximate outputs must be labeled and accompanied by error evidence.

## Beyond releases

Explore constrained choice graphs, continuous factors, image and signal transforms, scientific adapters, and crossed data/pipeline/model uncertainty. These are research directions, not promised APIs. [[Long-Term Architecture]] identifies the ownership boundaries such extensions would need.

Progress should be gated by a meaningful use case, a stated measurement target, and numerical validation. Avoid growing into a generic workflow optimizer whose central output is a best configuration.

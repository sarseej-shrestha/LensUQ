# Universe Model

A universe contains an integer ID, an immutable mapping of factor choices, and one sklearn Pipeline. The dataclass freezes metadata assignment, while the pipeline remains mutable because sklearn fitting creates state. Universe.from_steps clones every estimator; PipelineSpace always uses this path.

For each universe, preprocessing fits on X_train and the classifier fits on that universe's transformed X_train. The classifier is never borrowed from another representation. Predicting a held-out sample calls the already fitted pipeline without changing training statistics.

Independence must hold even when users pass the same estimator object twice, start with a fitted estimator, or configure copy=False. Cloning discards fitted state. Experiment additionally copies input data and labels per fit and input data per prediction, so an in-place scaler cannot alter the raw input seen by another universe. Tests check model coefficient arrays, preprocessing statistics, source objects, and repeated predictions.

These guarantees assume ordinary sklearn estimators obeying clone/fit semantics. A custom transformer that mutates global state is outside the isolation contract. Similarly, unseeded stochastic fitting can introduce variation unrelated to the declared choices. Users should set deterministic random states.

The fixed model family is a design constraint, not a claim that model uncertainty disappears. See [[Problem Statement]], [[Experiment Lifecycle]], and [[Open Problems]].

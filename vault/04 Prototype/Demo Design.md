# Demo Design

The bundled sklearn breast cancer dataset gives 569 numeric observations, 30 features, and binary labels without requiring a network download. Its lack of missing values makes imputation choices uninformative without a controlled perturbation.

Missingness is introduced artificially for the prototype to create a controlled preprocessing-sensitivity experiment. The demo is not a medical finding. It does not represent a clinical missing-data mechanism or validate clinical decision making.

A NumPy default_rng(42) independently masks each selected cell with probability 0.15. Selected features are mean radius, mean texture, mean perimeter, mean area, mean concavity, and mean concave points. The realized mask removes 532 of 3,414 eligible cells, or 15.58%; the other 24 columns and all labels remain unchanged.

Masking happens before a stratified 80/20 split with random_state=42, producing 455 training and 114 held-out rows. The mask uses no labels or fitted statistics. Every data-dependent preprocessing fit happens after splitting.

The space is mean / median / KNN(k=5) imputation crossed with standard / robust / min-max scaling. All nine universes use LogisticRegression(max_iter=5000, random_state=42), with other sklearn defaults unchanged. KNN distances use original feature units because imputation precedes scaling. The nominal universe is mean plus standard.

The demo prints the overall audit, ten most sensitive samples, and the complete explanation of the first ranked sample. No row is hardcoded. Class 1 is benign; class 0 is malignant. A reported probability is always for class 1.

Scaling changes the effective regularization geometry despite fixed model settings. A dominant scaling attribution is therefore plausible and is descriptive of this whole refitting procedure. See [[Prototype Results]] and [[Attribution Model]].

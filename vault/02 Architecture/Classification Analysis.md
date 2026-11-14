# Classification Analysis

Classifiers require `predict_proba`. Binary audits retain the positive-class DataFrame API and thresholded vote metrics. Multiclass probabilities have sample × universe × class shape in sklearn class order. Audits include nominal and majority classes, vote agreement and entropy, and each class's count, frequency, probability mean, population standard deviation, min, max, and range.

For multiclass attribution, the default analyzed quantity is each sample's majority-class probability. `class_index` selects one class for all samples. Binary attribution uses the second class probability. These are prediction distributions over declared pipelines, not calibrated error probabilities. See [[Attribution Model]].

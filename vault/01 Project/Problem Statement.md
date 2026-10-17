# Problem Statement

A conventional prediction fixes a preprocessing recipe and reports the output of the model fitted on that recipe's representation. In a workflow with several scientifically reasonable recipes, that output hides an unmeasured dependency.

Let P be a user-defined collection of preprocessing configurations. Each configuration p must train its own model on p(X_train), with preprocessing itself fitted only on training data. For a raw observation x, analyze the resulting predictions f_theta_p(p(x)). Reusing one model across representations answers a different, generally incoherent question.

## Assumptions

The same raw observations, training labels, split, model family, model settings, and seeded training policy are shared. Choices form a complete discrete factorial product in 0.1. Each level combination must be meaningful. Equal universe weights define the empirical measure being summarized.

## Claims and nonclaims

LensUQ quantifies dependence on the alternatives actually supplied. It does not certify that those alternatives are defensible, select the correct pipeline, infer calibrated intervals, or establish correctness. High agreement can accompany a shared error. Low agreement can reveal a consequential assumption without proving which decision is right.

Changing the space changes the estimand. Its justification belongs with the audit, not as an afterthought. [[Choice and Pipeline Space]] specifies the software contract; [[Pipeline-Induced Uncertainty]] explores the scientific question and [[Open Problems]] records unresolved cases.

# Future Research

## Proposed sequence of studies

Start with controlled synthetic datasets where missingness and decision boundaries can be varied independently. Measure how probability range, decision entropy, and factor attribution respond. Use known mechanisms to distinguish a correct numerical result from a meaningful scientific diagnostic.

Next, compare spaces specified independently by domain practitioners. Evaluate stability of sample rankings under reasonable additions or removals of levels. Record disagreements in space definition as a result rather than hiding them behind one curated grid.

Then cross preprocessing choices with repeated training seeds and data resamples. Estimate whether pipeline variation remains substantial after accounting for these other sources, and quantify interactions. Keep holdout selection separate from study design.

Finally, investigate constrained and sampled spaces. Compare approximate attributions with exhaustive small designs before claiming scalability. Report approximation error and the target measure explicitly.

## Evaluation principles

Use absolute probability variation and task-specific decision costs alongside normalized contributions. Include stable-but-wrong cases and high-interaction cases. Do not optimize demonstrations for dramatic disagreements or infer clinical usefulness from the built-in dataset.

Verify the literature tasks in [[Related Concepts]] before publication. [[Prototype Results]] establishes software behavior only. [[LensUQ 0.2]] handles the near-term metadata foundations; [[Long-Term Architecture]] records computational possibilities without committing to them.

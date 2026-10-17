# Measurement and interpretation

## Conditional finite-space variation

Fix a training partition, feature schema, classifier configuration, and seeded
training procedure. Declare a complete product of preprocessing alternatives.
For sample $x$, let $p_u$ denote the positive-class probability from universe $u$,
including the classifier fitted in that representation. Each of $U$ universes
has mass $1/U$. This empirical measure represents the specified alternatives;
it is not a posterior over pipelines and provides no coverage guarantee.

Adding, duplicating, or removing levels changes that measure and the resulting
attribution. Scientific defensibility is a domain judgment made before examining
the desired result. Choosing the space to maximize apparent disagreement would
change the question and risks a misleading demonstration.

## Probability and decision summaries

The nominal probability is $p_0$, the mean is $\bar p=U^{-1}\sum_u p_u$, and
population standard deviation is $\sqrt{U^{-1}\sum_u(p_u-\bar p)^2}$. The minimum,
maximum, and their difference describe the selected space's empirical extent.

At threshold $t$, $d_u=\mathbf{1}[p_u\geq t]$. Let $n_+=\sum_u d_u$,
$n_-=U-n_+$, and $q_+=n_+/U$, $q_-=n_-/U$. Then:

$$
A=\frac{\max(n_+,n_-)}{U},\qquad
C=\sum_u\mathbf{1}[d_u\ne d_0],\qquad
H=-q_+\log_2q_+-q_-\log_2q_-.
$$

Zero-log-zero terms are zero. $C$ counts disagreement with the nominal decision,
not the minority size or transitions between neighboring universe IDs. $H$ is
decision-vote entropy, not entropy of $\bar p$. An equal vote assigns the positive
majority label by convention, while agreement remains 0.5. Integer vote counts
are used before dividing, so complementary splits have exactly equal agreement.

Neither decision agreement nor a probability value is a correctness measure.
Unanimously wrong classifiers can have agreement one. `perfectly_stable` means
exactly equal probabilities in this finite space, not general robustness.

## Factorial main effects

For factor $F$, level $l$ appears in $n_l$ universes. Compute:

$$
SS_T=\sum_u(p_u-\bar p)^2,\qquad
SS_F=\sum_l n_l(\bar p_{F=l}-\bar p)^2.
$$

Under a complete balanced product with equal weights, the centered main-effect
subspaces are orthogonal. The remaining variation is
$SS_I=\max(0,SS_T-\sum_F SS_F)$. For $SS_T>0$, report $SS_F/SS_T$ and $SS_I/SS_T$.
These sum to one up to floating-point rounding. When predictions are constant,
all sums and contributions are zero. Small nonzero variation is not thresholded
away; its relative contributions should be interpreted alongside its absolute
range and total sum of squares.

For numerical stability, calculations first subtract the nominal probability
and then center those differences. This is algebraically identical and ensures
exactly constant inputs produce zero sums of squares. Residual clipping removes
tiny negative roundoff; it is not a replacement for validating the design.

For `[.1, .3, .5, .7]` in a two-by-two design (second factor varying fastest),
$SS_T=.20$, $SS_A=.16$, $SS_B=.04$: contributions are 80%, 20%, 0%.
For `[.1, .9, .9, .1]`, both main effects vanish and the residual is 100%.
For `[.25, .35, .55, .85]`, sums of squares are .16, .04, and .01, totaling .21.
These are regression tests, not fitted approximations.

The residual pools all orders of interactions. With only two nontrivial factors
it coincides with their interaction component, but the API deliberately retains
the aggregate name. A future model that identifies pairs must implement and
validate the appropriate decomposition explicitly. This is a descriptive ANOVA
partition, not an inferential ANOVA test: there are no p-values or noise estimates.

## Boundaries

Dependent choices, uneven weights, missing factorial cells, and sampled spaces
break the simple orthogonality argument and are rejected or outside scope.
The fixed model and data split isolate one source of variation operationally;
they do not prove a causal separation from training or model uncertainty.
Unseeded stochastic stages can contaminate the observed variation.

In the demo, changing scaling changes regularization geometry; changing
imputation changes both training representations and test inputs. Attribution
therefore measures each factor's effect on the refitted pipeline response.
KNN imputation precedes scaling and uses original units. Artificial independent
cell masking demonstrates the mechanism but does not model a clinical process.

Research context and citation-verification tasks live in the Obsidian vault's
Related Concepts and Open Problems notes. No novelty or medical inference is
claimed by this prototype.

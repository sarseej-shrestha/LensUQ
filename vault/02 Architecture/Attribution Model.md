# Attribution Model

The 0.1 estimator is a descriptive balanced-factorial sums-of-squares partition. It is not a hypothesis test and supplies no p-values.

For one observation, with grand mean bar(p):

$$
SS_T=\sum_u(p_u-\bar p)^2, \qquad
SS_F=\sum_l n_l(\bar p_{F=l}-\bar p)^2.
$$

Each level l of factor F occurs n_l times. Because all combinations occur once and all universes have equal weight, centered main effects are orthogonal. Report contribution_F = SS_F / SS_T and:

$$
SS_I=\max(0,SS_T-\sum_F SS_F), \qquad c_I=SS_I/SS_T.
$$

The interaction residual pools every higher-order interaction. It does not identify which pair or order is responsible. Zero main effects can coexist with strong interaction: [.1, .9, .9, .1] in a two-by-two grid has identical level means and a 100% residual.

For [.1, .3, .5, .7], main-effect SS values are .16 and .04, total .20, with no residual. For [.25, .35, .55, .85], the main effects remain .16 and .04 and residual SS is .01. Tests enforce these exact analytic patterns and conservation across larger products.

Exactly constant predictions receive zero contributions and a perfect-stability flag. Small nonconstant variation is retained, so large relative contributions must be read alongside absolute range and total SS. Calculations center differences from the nominal prediction to avoid cancellation and constant-input roundoff. Nonnegative residual clipping addresses floating-point noise only.

Incomplete or duplicated cells are rejected. Uneven weights, correlated factors, failed universes, and sampled spaces require another estimator; see [[Open Problems]] and [[Long-Term Architecture]]. Level means are provided so users can inspect direction as well as fractional size. These are effects on the full refitted pipeline, not isolated causal transformer effects.

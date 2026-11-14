# Attribution Model

On a complete equally weighted Cartesian design, total SS is the sum of squared deviations from the grand mean. For factor F, main SS is `sum_l n_l (mean_l - grand_mean)^2`. For pair A,B, pair SS is `sum_ab n_ab (mean_ab - mean_a - mean_b + grand_mean)^2`. The remainder after all main and pairwise effects is `interaction_residual`, containing higher-order interactions. Fractions sum to approximately one for nonconstant outputs; exactly constant outputs receive zero fractions.

The decomposition is descriptive. It is not a hypothesis test or a causal effect of one transformer. The model is refitted in every universe. Incomplete sampled designs have no attribution in 0.2. See [[Execution Strategies]], [[Classification Analysis]], and [[Regression Analysis]].

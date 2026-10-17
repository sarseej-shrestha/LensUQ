# Uncertainty Metrics

For U equally weighted universe probabilities p_u, report nominal p_0, mean, population standard deviation (ddof=0), minimum, maximum, and range. These summarize a finite declared space and are not confidence intervals.

At threshold t, let d_u = 1[p_u >= t], n_+ = sum(d_u), n_- = U - n_+, and q_+ = n_+/U, q_- = n_-/U.

$$
A = \max(n_+, n_-)/U
$$

Agreement A is 1 for unanimity and 0.5 for an even split. Compute it from integer counts before division to preserve exact equality between complementary vote patterns. The majority label uses the positive class in an even tie; this is only a reporting convention.

$$
C = \sum_u 1[d_u \ne d_0], \qquad
H = -q_+\log_2 q_+ - q_-\log_2 q_-.
$$

C counts flips relative to nominal, which can exceed the minority count. H is entropy of thresholded decisions in bits, using zero for zero-log-zero terms. It is different from entropy of a model probability or of the mean probability.

Perfect stability requires exactly equal probabilities. Decision unanimity alone is weaker: substantial probability differences can sit on the same side of the threshold. Reporting both views avoids hiding that difference.

Sensitivity ranking is lexicographic: agreement ascending, range descending, standard deviation descending, then audit input order for exact ties. It prioritizes decision instability, so a narrower range that crosses the threshold can rank above a wider range without class changes.

[[Attribution Model]] explains probability variation, independently of threshold. [[Prototype Results]] records actual values. [[Testing Strategy]] includes hand-computed metric oracles.

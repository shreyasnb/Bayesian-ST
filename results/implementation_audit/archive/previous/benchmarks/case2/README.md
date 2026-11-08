# Field references

The full-observation `GROUSE` implementation uses the rank-one Grassmann geodesic
update for one-half squared projection residual, as in Balzano, Nowak and Recht,
*Online identification and tracking of subspaces from highly incomplete
information*, Allerton 2010, https://arxiv.org/abs/1006.4046. The complete-data
specialization is appropriate because the draft assumes all vector components
are observed. It is not a missing-data implementation.

`EWCovariancePCA` estimates the leading eigenspace of the exponentially weighted,
uncentered sample covariance. With a scalar marginal Bingham increment,
identity initial parameter and matching forgetting, its modal eigenspace agrees
with the Bingham modal estimate. This equivalence is an informative control;
posterior moments affect downstream decisions rather than creating a new modal
tracking advantage in that regime. `RollingPCA` is the corresponding finite
window reference. `reconstruct` uses the same known Wiener shrinkage as the
matched Bingham reconstruction, so its comparison isolates the subspace estimate.

Tune forgetting, window length and GROUSE step size on independent validation
trajectories with equal candidate counts, rank and observation budgets. Reuse
selected settings on all held-out test trajectories. Known calibration vectors
must not be given to passive references. Calibration therefore compares the
conditional Bingham mean and mode using their identical parameter sequence.

GREAT (Sasfi et al., https://arxiv.org/abs/2412.09052v2) and GeRoST (Bharadwaj
et al., https://arxiv.org/abs/2604.00825v2) are relevant extensions. They are not
included under those names: their precise update rules, robust radii and tuning
requirements need separate source verification, and a fabricated implementation
would not be a valid benchmark. The selected GROUSE and covariance comparisons
are explicit, reproducible algorithms suitable for the present fully observed
protocol.

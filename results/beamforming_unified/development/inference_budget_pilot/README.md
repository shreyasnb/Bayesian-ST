# Inference budget engineering

These four targets use development-only roots 6571099/6571199, independently of quick, validation and paper data. Only runtime and MCMC diagnostics were examined; decision outcomes were not inspected. Budgets were 3 chains, 256 burn-in sweeps and 512 retained draws, with no outcome-driven extension.

Stationary scalar variances have ESS 190–417 and split Rhat at most 1.003 in these fixtures. The strict overall flag nevertheless fails because the largest between-chain discrepancy among every monitored state-component tail is 20–29% of its width. The final implementation retains that same strict flag and adds separate scalar, state-mean, full-state-tail, and joint forecast-covariance-mean checks. This distinguishes the precision of a scalar or mean functional from unverified full-state tails without relaxing a threshold.

Moving process concentration is genuinely unresolved at this budget: the matched fixture has minimum scalar ESS 5.9 and maximum Rhat 1.66; the ULA fixture has kappa ESS 9.9. The declared paper pilot retains the bounded budget and flags. It does not claim reliable process-concentration recovery merely because the conditional sampler runs.

The physical ULA trajectory has no true kappa. Its assumed isotropic transition can be assessed prospectively, but cannot be called the data-generating motion law.

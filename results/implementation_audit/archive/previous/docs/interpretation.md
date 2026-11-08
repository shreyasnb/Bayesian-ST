# What the experiments support

The framework is most compelling when the desired output is a decision that
depends on an uncertain subspace, rather than only a subspace basis. The new
notebook figures separate these questions and show losses, ties, and costs.

## Original application studies

The field pilot has eight independent trajectories per scenario. At the last
observation, mean reconstruction MSE is about 61% below the selected GROUSE
reference under unequal powers and 37% below it under sparse corruption. Most
of that difference is already present in EW PCA/the modal estimate. The
posterior mean's additional reductions relative to its own mode are about
2.2% and 1.1%, with paired intervals crossing zero. Matched passive
reconstruction has essentially no mean–mode difference. Rolling PCA remains
competitive. These results do not isolate a large Bayesian-averaging advantage
for the original passive application.

The array pilot shows a modest mean–mode SINR gain under drift (about 0.107 dB
at the terminal time), and about 0.956 dB versus the tuned loaded-covariance
reference in that same scenario. Under desired-signal contamination, loaded
covariance is much better than both Bingham decisions. Such outcomes are
important model-mismatch tests, not observations to remove from a figure.

With uniform initialization, fixed variances, and matching forgetting, the
marginal Bingham mode and EW PCA coincide. Improvements over GROUSE alone
therefore do not establish that posterior uncertainty is responsible.
The heatmaps include a same-posterior modal control to make this distinction.

## Controlled low-data decisions

The added experiment uses a stationary real rank-one subspace in three
dimensions, sampled from the uniform prior, and the stated Gaussian observation
model. Every method receives the same observations and known noise levels.
Fresh query/steering vectors are independent of the hidden state and available
to every method at decision time. There are 400 independent test trajectories,
160 independent validation trajectories, and five predeclared snapshot counts.

GROUSE, scalar-shrunk PCA, and loaded covariance use nine validation candidates
per applicable task/horizon/noise regime. The ranges include no shrinkage and
a near-isotropic covariance limit. They were broadened after validation picked
old grid boundaries; no test outcomes select settings. The tuned families,
selected boundaries, and every regime remain in the saved tables. This compares
a constant-step GROUSE family, not every possible GROUSE schedule or tracker.

At signal variance 1, noise variance 0.2, and four snapshots:

| Task | Mean loss | Reference loss | Reduction |
| --- | ---: | ---: | ---: |
| Fresh response versus mode/EW PCA | 0.09784 | 0.12738 | 23.19% |
| Fresh response versus shrunk PCA | 0.09784 | 0.10915 | 10.36% |
| Reconstruction versus mode/EW PCA | 0.27189 | 0.30392 | 10.54% |
| Reconstruction versus shrunk PCA | 0.27189 | 0.28729 | 5.36% |
| Beamformer power versus mode | 0.41912 | 0.44401 | 5.61% |
| Beamformer power versus loaded covariance | 0.41912 | 0.42888 | 2.28% |

All six paired gain intervals exclude zero. Intervals are across independent
trajectories; eight query directions within a trajectory are averaged rather
than counted as independent trials. Percent displays scale the paired mean
loss difference and its interval by the observed reference mean; they are not
confidence intervals for a random loss ratio or simultaneous guarantees.

This is evidence for a particular mechanism: averaging orientations improves
predictions and covariance-based decisions when the orientation is uncertain.
The better comparison is with tuned scalar shrinkage, which already captures
much of the benefit. The remaining gain reflects adaptation to the posterior
spectrum and directional structure, rather than the use of shrinkage alone.

The gains shrink as information grows. At sixteen moderate-noise snapshots,
fresh-response and beamforming differences versus tuned shrinkage have
intervals containing zero. Under weak signal, a strongly loaded covariance is
also competitive: several apparent beamforming advantages disappear when the
loading grid includes the near-isotropic limit. These retained controls prevent
turning a limited tuning grid into an artificial Bayesian advantage.

The experiment is a diagnostic, real low-dimensional setting. It does not
establish the same effect size for the 16-channel complex array or drifting
rank-three field problem, nor general robustness to non-Gaussian observations.
The exact posterior is optimal for the specified prior/likelihood and loss;
that theoretical fact is established literature. The simulation quantifies
its practical size against explicit controls.

## Uncertainty and cost

The 4,000 controlled posterior means agree with a doubled quadrature order to
within 1.66e-9. Production decision timing excludes reference-order audits and
coverage integration. The controlled, correctly specified model gives 93–94
covered states out of 100 for nominal 95% regions; Wilson intervals contain
95%. The binned risk plots compare the posterior risk with observed losses.
These checks concern the joint prior-generating model, not uniform coverage
for every fixed unknown state.

The drifting field application's nominal regions have 0/8 terminal coverage
in its matched-covariance setting. This remains a serious limitation of the
approximate prediction/forgetting protocol. Exact posterior mass, model-based
risk, and frequentist calibration under a misspecified motion law are different
claims. The results do not justify safety guarantees based on those regions.
Sampled high-dimensional EM ascent also remains unverified; the positive
uncertainty term in its residual statistic is the supported noise-update insight.

For the original implementations, terminal cumulative median work is roughly
43 s for the array posterior-mean decision versus 0.022 s for the mode, and
5.3 s for the field mean versus 0.012 s for GROUSE. These numbers include three
posterior reporting times through 72 updates and are machine-dependent
reference-MCMC timings. They are not per-step latency or lower bounds on the
cost of Bayesian inference. In the controlled three-dimensional experiment,
validated deterministic moments make the cost far smaller in absolute terms.

A practical use of the framework is to retain the parameter recursion and
request moments, directional bounds, or noise fits at the times when those
quantities affect a decision. That policy is a deployment choice, not a
performance result demonstrated by the current experiments. Faster validated
moment evaluators and calibrated state models are important further work.

If only a fast basis estimate is needed, the present evidence favors using a
suitable cheap tracker. If response accuracy under weak information, directional
risk, posterior task bounds, or uncertainty-aware noise diagnostics matter,
the framework provides coherent quantities that a basis-only GROUSE output
does not itself supply. Other trackers can be augmented with uncertainty;
that would be a separate, worthwhile comparison.

# Independent mathematical audit

Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/audit_beamforming_independent.py > results/beamforming_unified/development/independent_audit.jsonl`.

This uses development seed 81536029. It does not inspect formal validation or test outcomes, select a favorable regime, or require any method to win. The independent references include rescaled tilted-Beta integrals for repeated eigenvalues, a Gaussian covariance-likelihood KL Hessian, direct ordered-prior integration, transition normalizers, and boundary-layer CP1 CDFs through spectral span one million. Aligned/orthogonal CDFs use independent analytic values; nonaligned sharp CDFs are compared with fixed-budget exact iid draws. Monte Carlo moment checks use explicit fixed budgets; they are not deterministic integration certificates.

The tied-top `n=16`, multiplicity seven, span 100 sampler fixture deliberately retains the default proposal-budget limit. The best implemented envelope accepts about 0.0008 of proposals, so 8,192,000 proposals do not deliver 8,192 requested draws. Its `sampling_ok=false` is an explicit unresolved efficiency limit, not an error in the rejection law: observed acceptance agrees with the independent normalizer, and the deterministic moments remain accurate. This is not evidence that ordinary simple-mode array posteriors fail. No partial draw set is used for scientific inference.

The response check verifies that the trace of the full response covariance equals the scalar vector risk `v^H(M-M^2)v`. The full covariance itself differs from `M-M^2`, so it is not recoverable from that risk expression.

The manifest records exact audited sources and the final raw-output hash. A completed audit does not establish Markov-chain mixing, exact empirical credible mass, simultaneous frequentist coverage, or posterior superiority over a baseline.

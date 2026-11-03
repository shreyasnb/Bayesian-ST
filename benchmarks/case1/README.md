# Array references

`LoadedCovariance` forms an exponentially weighted, uncentered sample covariance
with bias correction for the accumulated weight and relative diagonal loading.
`mvdr` implements the standard distortionless minimum-variance solution; see
Van Trees, *Optimum Array Processing*, 2002, Chapters 6–7. Supplying the true
covariance gives an oracle for evaluation only. `sample_output_power` expects
independent, row-wise evaluation snapshots and never updates an estimator.

Select the loading and forgetting factor using an independent validation
trajectory, with the same candidate count and trajectory lengths as competing
methods. Never tune using test interference covariances or evaluation snapshots.
Compare posterior-mean and modal Bingham covariances using the same fitted
variances and parameter sequence; otherwise the difference cannot be attributed
to uncertainty integration. All training snapshots are interference-only in the
matched model. Desired-signal contamination belongs in a labelled mismatch run.

The complex Bingham filter is implemented by the common library. This reference
folder does not claim to reproduce the algorithm of Sabzikar et al. (2017),
*Subspace tracking in a sensor array using complex Bingham distribution*,
DOI: 10.1049/iet-rsn.2017.0206. Its transition model and numerical implementation
would need separate verification before an author-labelled comparison. The
reproducible covariance reference directly tests the task contribution without
making that unsupported attribution.

# Stationary beamforming at a fixed measurement budget

The current stationary comparison uses **200 measurements for every sensor
count**, with $n=32,64,128$. The primary results use $t=T=200$; saved
checkpoints at $t=25,50,100,200$ support the supplementary comparisons.
The previous $t=n$ cohort remains in its original files and archive. It is not
relabelled as a fixed-200-measurement experiment.

The interference direction is fixed, with $|a^{\mathrm H}h|^2=1/2$,
$\sigma^2=0.2$ and $\tau^2=1$. The Bingham tracker starts with the uniform
prior $A_0=0$ and uses $\beta=1$. All ten implemented methods remain in the raw
cohort and machine-readable summaries. The main table shows the posterior mean,
mode, GREAT with a sliding window, and regularized sample covariance; a
consolidated supplementary table shows those same columns at all three sensor
counts.

There are 64 independent validation simulations and 256 independent test
simulations per sensor count: 192 validation and 768 final simulations in
total. Candidate grids and the measured-data validation
criterion are unchanged: settings are selected using additional independent
array measurements, without using the true covariance to choose settings.
The selected settings and their source configuration are recorded by the
fixed-$T$ cohort producer. The exact covariance is used only for evaluation.

The paper-stage stream seeds retain their existing construction:
`SeedSequence([base_seed, n, condition_index, trial])`, with validation base
`2026111001`, test base `2026111002`, static condition index zero and simulation
identifiers `0,...,255` for testing. The stream generator obtains its random
stream, initial point-reference subspace and independent evaluation stream
from distinct child sequences. All methods in one simulation use the same
observations. Increasing the stationary horizon does not change these seed
conventions or turn additional measurements into independent repetitions.

## Quantity and averaging

For each test simulation, the recorded exact covariance gives

$$
\ell_{200}=10\log_{10}\!\left(
 \frac{w_{200}^{\mathrm H}R_{200}w_{200}}{p_{200}^{\star}}\right),
\qquad
p_{200}^{\star}=(a^{\mathrm H}R_{200}^{-1}a)^{-1}.
$$

`true_ratio_db` stores this quantity. Each reported mean is the arithmetic
mean of the 256 individual dB values. It is not the logarithm of an average
linear power ratio. Two-sided Student 95% confidence intervals use the sample
standard deviation, standard error $s/\sqrt{256}$ and 255 degrees of freedom.
Smaller values indicate lower output power relative to the known-covariance
minimum under the unit-gain constraint.

Paired summaries subtract the posterior-mean score from each other method's
score on the same simulation identifier. Positive differences favor the
posterior mean. All nine references, including ties or losses, remain in the
paired summary and per-simulation paired records. These are pointwise intervals,
without a simultaneous-coverage claim.

## Completed results

The following values are the arithmetic mean relative power and its pointwise
95% confidence interval, in dB. These values come from the completed
fixed-$T=200$ cohort, with 256 independent final simulations in each group.

| Sensors | Posterior mean | Mode | GREAT (sliding window) | Regularized sample covariance |
| --- | --- | --- | --- | --- |
| 32 | 0.118789 [0.115119, 0.122459] | 0.114388 [0.110875, 0.117901] | 0.363181 [0.351389, 0.374973] | 0.378834 [0.369074, 0.388595] |
| 64 | 0.236885 [0.231411, 0.242358] | 0.222701 [0.217619, 0.227782] | 0.349989 [0.341957, 0.358021] | 0.576354 [0.565182, 0.587526] |
| 128 | 0.481558 [0.473464, 0.489652] | 0.440108 [0.432650, 0.447566] | 0.677332 [0.665888, 0.688776] | 0.856078 [0.843742, 0.868414] |

At this fixed measurement budget, the mode has lower mean relative power than
the posterior mean at all three sensor counts. Both have lower mean relative
power than the sliding-window GREAT and regularized covariance references
shown here. This is not a general advantage over every GREAT implementation:
exponentially weighted GREAT is close to the mode, including 0.222710 dB at
$n=64$ compared with 0.222701 dB for the mode. All ten methods and their paired
comparisons remain in the machine-readable exports.

The completed raw data contain 30,720 checkpoint records
($768\times4\times10$), including 7,680 records at the primary checkpoint.
All method records passed their numerical checks. The 3,072 native mean
evaluations had no unresolved results, maximum reported mean error bound
$9.99904\times10^{-11}$, maximum series order 2,510, and no cap extensions.
The maximum unit-gain error was $4.45311\times10^{-16}$. These checks concern the
native rank-one mean and action calculations; they do not certify other
posterior functionals.

The stationary dimension comparison is supplementary:
`results/supplementary/beamforming/stationary_dimensions_T200.pdf`.
The main stationary results retain the $n=64$ table below and
`results/main/beamforming/task_performance_output_power.pdf`, which shows the
fixed-$T=200$ stationary comparison. The consolidated supplementary table
retains all three sensor counts.

Moving interference is presented independently in
`notebooks/beamforming_moving_interference.ipynb`, with its own result root,
`results/beamforming_moving/`. Its optional candidate main figures,
`moving_interference_smooth_output_power.pdf` and
`moving_interference_abrupt_output_power.pdf`, do not automatically replace
the stationary main figure. This fixed-budget stationary run did not
regenerate the moving, block, variance, contact, theory, or computational
experiments.

## Run provenance

The production manifest records the actual producer source hashes and binds
the fresh validation certificate at
`results/beamforming/data/validation/stationary_T200/validation.json`.
That certificate reports 319 passing tests, core hash
`401911374ba2f2782f7cbb36bd783c174a5d059bc4a6df7c1a9a2a41f6dcc147`,
and scientific source hash
`e926a6a16fea47e35f0625afcc811d5fbdd533fb812c8c1fa9314484f16b24ee`.
The configuration's `source_commit`,
`a91faabeb10a63b1d1c613d75cc32503cfc0cac6`, is the inherited historical base;
the manifest's source hashes identify the implementation used for the new
run. It is not a claim that the new run used only files at that old commit.

## Saved files and export

The raw fixed-budget results and enclosing producer manifest are:

- `results/beamforming/data/paper/performance/stationary_T200/stationary.csv`
- `results/beamforming/data/paper/performance/stationary_T200/manifest.json`

After that cohort is completed, run:

```bash
python scripts/export_stationary_beamforming_tables.py
```

The exporter checks all four checkpoints, all ten methods, finite exact powers,
successful numerical flags and the same 256 test identifiers in every group.
It verifies the raw file against its producer manifest before writing any
tables. It does not run an experiment, retune settings or modify the cohort
manifest.

The manuscript tables are written to:

- `results/main/beamforming/stationary_T200_table.tex`: $n=64$, $T=200$.
- `results/supplementary/beamforming/stationary_T200_table.tex`: consolidated
  $n=32,64,128$, always $T=200$.

Both tables have four method columns and explicit rows for the mean relative
power and its 95% confidence interval, in dB. They use standard `booktabs`
commands for inclusion in the manuscript. Both compile with `article` at
10 pt, `geometry` with a 6-inch text width, `booktabs`, and `amsmath`, without
overfull boxes. Method headers wrap within their columns; no resizing is
needed.

The complete presentation summaries are kept separately under
`results/beamforming/tables/paper/performance/stationary_T200/`:

- `summary.csv`: all ten methods at every sensor count, at $t=200$.
- `paired_summary.csv`: all nine paired references at every sensor count.
- `paired_trials.csv`: every paired simulation difference used by those summaries.
- `summary.json`: full-precision summaries, raw-data/configuration hashes,
  averaging definition, source configuration and output hashes.

The original stationary summary tables and measured validation outcomes remain
unchanged. The new tables summarize the newly produced fixed-$T$ raw records;
there is no fallback to the earlier measured-power summaries. The production
cohort manifest remains immutable and covers its original producer outputs.
The separate derived report, `summary.json`, binds the raw CSV hash, stable
configuration hash, exporter source hash, and hashes of the five generated
table assets (two TeX files and three CSV files). This preserves the production
manifest while making the derived exports independently verifiable.

"""Bind source-verified extension evidence to the gate and report every outcome."""
from pathlib import Path
import hashlib
import json
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lib.validation import require_gate

OUT=ROOT/'results/dev_prediction'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_run(directory,metadata_name):
    path=OUT/directory;metadata=json.loads((path/metadata_name).read_text())
    sources={**metadata.get('sources',{}),**metadata.get('postprocessing_sources',{})}
    for name,digest in sources.items():
        if sha(ROOT/name)!=digest:raise RuntimeError(f'Stale numerical source: {directory}/{name}')
    for name,digest in metadata['tables'].items():
        if sha(path/name)!=digest:raise RuntimeError(f'Changed result table: {directory}/{name}')
    return metadata


def empirical(low,high):
    return 'implemented and checked' if low>0 else 'empirically unsuccessful' if high<0 else 'empirically inconclusive'


def main():
    gate=require_gate(ROOT,gate_path=OUT/'validation.json')
    motion=checked_run('motion','config.json')
    applications=checked_run('applications','config.json')
    process=checked_run('process','status.json')
    # The only core revision after the initial study gate was a certificate-path
    # correction plus its regression. Every numerical study source is unchanged
    # and verified above. Preserve captured run certificates in their metadata.
    binding=dict(final_core_hash=gate['core_hash'],captured_run_core_hashes={
        'motion':motion.get('core_hash'),'applications':applications.get('core_hash')},
        verified_sources={name:{**meta['sources'],**meta.get('postprocessing_sources',{})} for name,meta in
                          [('motion',motion),('applications',applications),('process',process)]},
        verified_tables={name:meta['tables'] for name,meta in
                         [('motion',motion),('applications',applications),('process',process)]},
        scope='Source-verified association with final gate; no new Monte Carlo replication',
        change='Relative-root certificate lookup and its regression test; scientific study dependencies unchanged')
    (OUT/'gate_binding.json').write_text(json.dumps(binding,indent=2))
    paired=pd.read_csv(OUT/'motion/paired.csv')
    primary=paired[paired.primary.astype(bool)].iloc[0]
    motion_status=empirical(primary.low,primary.high)
    application_pairs=pd.read_csv(OUT/'applications/paired.csv')
    tuning=pd.read_csv(OUT/'applications/tuning.csv')
    app_records=[]
    for application,scenario in [('array','Drift'),('field','Matched')]:
        row=application_pairs[(application_pairs.application==application)&
                              (application_pairs.scenario==scenario)&
                              (application_pairs.reference=='Shrunk PCA')].iloc[0]
        selected=tuning[(tuning.application==application)&tuning.selected.astype(bool)&
                        (tuning.method=='Posterior mean')]
        selected_failures=int(selected.sampling_failures.sum())
        failures=int(row.posterior_sampling_failures)
        state='unresolved' if failures or selected_failures else empirical(row.gain_low,row.gain_high)
        if 'unresolved' in str(row.interpretation).lower():state='unresolved'
        app_records.append(dict(application=application,status=state,selected_validation_failures=selected_failures,
                                test_failures=failures,paired_gain=float(row.gain),low=float(row.gain_low),
                                high=float(row.gain_high),interpretation=str(row.interpretation)))
    statuses=[
        dict(checkpoint=0,status='implemented and checked',scope='Baseline tests, likelihood identities, equivariance, hashes and preserved outputs',artifacts='baseline/audit.json; validation.json'),
        dict(checkpoint=1,status='implemented and checked',scope='Explicit identity, partial, exact and causal motion; sensor-fixed and co-rotating inputs',artifacts='motion/preregistered.json; applications/predeclared_config.json'),
        dict(checkpoint=2,status='implemented and checked',scope='Exponential memory and rolling likelihood queue; independent mixture validation',artifacts='motion/tuning.csv; motion/selection.json; motion/paired.csv'),
        dict(checkpoint=3,status=motion_status,scope='Geometry and causality checked; primary causal-versus-tuned-memory holdout outcome',artifacts='motion/paired.csv; motion/paths.csv; motion/cost.csv'),
        dict(checkpoint=4,status='implemented and checked',scope='Rank/codimension-one transition moments and inverse projection; independent real-circle full-density reference',artifacts='process/reference_checks.json; process/coverage_trials.csv'),
        dict(checkpoint=4,status='unresolved',scope='General-rank stochastic moment inversion and calibrated uncertainty; explicitly rejected',artifacts='process/status.json'),
        dict(checkpoint=5,status='implemented and checked',scope='Matched-process 200-trial calibration per kappa; four distinct coverages, widths, innovations and separate noise ablations',artifacts='process/coverage_summary.csv; motion/coverage_paired_noise.csv; applications/noise.csv'),
        *[dict(checkpoint=6,status=r['status'],scope=r['application']+' primary application versus scalar shrinkage',artifacts='applications/paired.csv; applications/tuning.csv') for r in app_records],
        dict(checkpoint=7,status='implemented and checked',scope='Checkpoint handoff, sources, table checksums, commands, manuscript notes and measured cost',artifacts='manifest.json; gate_binding.json; README.md'),
    ]
    pd.DataFrame(statuses).to_csv(OUT/'checkpoint_status.csv',index=False)
    summary=dict(validation=gate,checkpoint_status=statuses,motion_primary=primary.to_dict(),
                 selected_motion_policy=motion['selected'],motion_checks=motion['checks'],
                 applications=app_records,application_checks=applications['checks'],
                 process_reference=json.loads((OUT/'process/reference_checks.json').read_text()),
                 process_status=process['status'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Development checkpoint findings','',
           'Branch: dev. Reviewed baseline: `57f520cbe860b73e13d71bae5c31391e2a009482`.',
           'Original artifacts are preserved; this directory contains the extension only.',
           'The baseline audit records the guide hash, software versions and original budgets.',
           '', '| Checkpoint | Status | Scope | Evidence |', '| --- | --- | --- | --- |']
    for row in statuses:
        links='; '.join(f'[{name}]({name})' for name in row['artifacts'].split('; '))
        lines.append(f'| {row["checkpoint"]} | {row["status"]} | {row["scope"]} | {links} |')
    lines.extend(['', '## Memory and velocity','',
        f'The common validation policy is beta={motion["selected"]["beta"]}, W={motion["selected"]["window"]}, lag={motion["selected"]["lag"]}, damping={motion["selected"]["damping"]}.',
        'It is shared across test laws and input frames; no hidden test speed selects a policy.',
        'The lag grid was expanded using validation only. Its remaining boundary flag is retained.',
        f'The preselected fast-drift, sensor-fixed primary gain (tuned forgetting minus causal velocity) is {primary.mean_gain:.5f}, with paired 95% interval [{primary.low:.5f}, {primary.high:.5f}].',
        'See [paired losses](motion/paired.csv) and [all motion findings](motion/findings.md), including transients, acceleration, turns, weak excitation, low SNR and corruption.',
        'Learned rotations and rolling windows receive no deterministic tracking certificate.',
        f'Posterior/prior pilot sampling failures: {motion["checks"]["posterior_sampling_failures"]}/{motion["checks"]["prior_sampling_failures"]}.',
        'Only eight trajectories per conditional posterior regime are evaluated; Wilson intervals are wide.',
        'Final response coverage uses method-shared held-out noise in motion/coverage_paired_noise.csv.',
        'The raw inherited method-specific query evaluation is preserved with an explicit postprocessing audit.',
        '', '## Applications','',
        'One task-loss validation policy per application is frozen before independent tests.',
        'Loading and memory form a Cartesian grid; scalar shrinkage, same-posterior mode,',
        'EW PCA and the appropriate application references are included. Field GROUSE and',
        'rolling-memory grids extend the earlier boundaries. Extra regimes are exploratory.',
        '', '| Primary application vs shrinkage | Status | Selected validation failures | Test failures | Paired gain interval |',
        '| --- | --- | ---: | ---: | --- |'])
    for row in app_records:
        lines.append(f'| {row["application"]} | {row["status"]} | {row["selected_validation_failures"]} | {row["test_failures"]} | [{row["low"]:.5f}, {row["high"]:.5f}] |')
    lines.extend(['', 'Intervals with capped numerical failures do not support positive performance claims.',
        'A negative or inconclusive comparison stays visible. Full tuning and trial-level',
        'diagnostics are in applications/; point estimates do not repair failed numerical checks.',
        '', '## Process uncertainty','',
        'The explicit isotropic transition is tested against an independent full-density real Gr(1,2) reference.',
        'Kappa is a known simulation parameter. Fixed beta=0.85 controls are diagnostics, not tuned application competitors.',
        'The predicted moment is exact conditional on the maintained preceding Bingham law; the density projection is approximate.',
        'A reference fixture has inverse residual near 4e-17 and predicted-density TV near 0.0283. Those measure different errors.',
        'General-rank inversion remains unresolved; it is not replaced by scalar rescaling.',
        'See [reference checks](process/reference_checks.json) and [coverage with widths](process/coverage_summary.csv).',
        '', '| Matched kappa | Method | Ball coverage | HPD coverage | Clean coverage | Noisy coverage | Trials |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |'])
    process_summary=pd.read_csv(OUT/'process/coverage_summary.csv')
    matched=process_summary[process_summary.law=='Matched transition']
    for row in matched.itertuples():
        if row.method not in ['Projected transition','Full-density reference']:continue
        lines.append(f'| {row.kappa:g} | {row.method} | {row.ball_coverage:.3f} | {row.hpd_coverage:.3f} | {row.clean_coverage:.3f} | {row.noisy_coverage:.3f} | {row.trials} |')
    lines.extend(['', 'Coverage intervals are pointwise Wilson intervals; favorable finite results are not a general calibration theorem.',
        'Deterministic drift and observation-mismatch failures remain in [process findings](process/findings.md).',
        'Observation contamination produces undercoverage even for the full-density filter.',
        'The two noise-learning ablations check frozen-prior objectives, not full state-space likelihood ascent.',
        '', '## Cost and figures','',
        'motion/cost.csv and applications/metrics.csv separate prediction, update, moment and decision costs.',
        'Process inversion is nested within prediction time; embedded moment costs are missing rather than falsely zero.',
        'Reporting frequencies, worker counts and software versions are in the corresponding configurations.',
        'manifest.json records hardware and notes that cloud wall-clock timings can overlap other studies.',
        'Figures use TeX and 200 dpi. Main figures show motion error and uncertainty, paired application losses,',
        'and coverage versus width. Extra tuning curves, conditions and runtime detail are supplementary.',
        '[Motion and uncertainty](figures/motion.pdf), [paired application losses](figures/application_gains.pdf),',
        'and [matched-process coverage and width](figures/process_coverage_width.pdf) are the three main figures.',
        '', '## Commands','', 'Run from the repository root with the documented Python/TeX dependencies.',
        'Verify the delivered evidence and regenerate its figures:', '', '```bash',
        'python scripts/summarize_prediction_ablation.py',
        'python scripts/plot_prediction_ablation.py',
        'python scripts/write_dev_manifest.py', '```', '',
        'Recompute the numerical studies in fresh directories using the frozen scientific settings.',
        'The same seeds reproduce the experiment; they do not create extra independent evidence:', '', '```bash',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/validate_dev.py',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_prediction_ablation.py --workers 2 --output-dir results/dev_prediction/reproduced_motion',
        'python scripts/check_motion_paired_noise.py --output-dir results/dev_prediction/reproduced_motion',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/check_process_prediction.py --output results/dev_prediction/reproduced_process',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_application_ablation.py --workers 2 --output-dir results/dev_prediction/reproduced_applications',
        '```', '',
        'The captured run gates remain in original run metadata. gate_binding.json verifies every',
        'numerical source/table before associating those runs with the final gate after the certificate-path',
        'fix and regression test. This is dependency-verified reuse, not additional independent simulation.',
        'The motion reproduction command uses a fresh directory: the original producer deliberately',
        'rejects its captured old gate rather than silently rewriting completed-run metadata.',
        'See [development documentation](../../docs/development.md) and [manuscript notes](../../docs/dev_manuscript_notes.md).',''])
    (OUT/'README.md').write_text('\n'.join(lines))
    print(json.dumps(dict(motion=motion_status,applications=app_records),indent=2))


if __name__=='__main__':main()

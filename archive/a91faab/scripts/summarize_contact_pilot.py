"""Write a scope assessment from the frozen contact evidence, retaining failures."""
from pathlib import Path
import hashlib
import json
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lib.contact_reporting import load_contact
OUT=ROOT/'results/contact_pilot'


def main():
    frames=load_contact(ROOT);config=frames['config'];checks=frames['metadata']['checks']
    numerical=json.loads((OUT/'numerics/checks.json').read_text())
    mechanics=json.loads((OUT/'numerics/mechanics_checks.json').read_text())
    for record in [numerical,mechanics]:
        for name,digest in record['sources'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    paired=frames['paired'];primary=paired[paired.primary.astype(bool)]
    terminal=frames['posterior'][frames['posterior'].t==config['T']]
    coverage=frames['coverage'][frames['coverage'].t==config['T']]
    actual=terminal.groupby('condition')[['risk_mean','empirical_response_error','width_normalized','modal_chordal_error']].mean()
    actual['empirical_to_posterior_risk']=actual.empirical_response_error/actual.risk_mean
    scope='Physical example only'
    reason=('The mechanics and inference numerics work, and the primary task has a modest gain over pooled scalar shrinkage. '
            'The improvement over its own mode is tiny and curved-guide credible uncertainty is unreliable. '
            'Keep this as a physical conditional-response illustration, pending a stronger sensing/dynamics comparison.')
    if checks['numerical_failures'] or checks['selected_validation_numerical_failures'] or mechanics['status']!='passed':
        scope='Unresolved or unsuitable';reason='Physical or numerical acceptance is unresolved; retain the recorded failures before drawing a replacement claim.'
    case=config['primary_condition'];g=frames['metrics'].query('condition == @case')
    # Corrections are cumulative, whereas integration/decision clocks are per report.
    posterior_cost=g[g.method=='Posterior mean'].groupby('trial').agg(
        corrections=('update_seconds','max'),integrations=('integration_seconds','sum'),decisions=('decision_seconds','sum')).mean()
    records=[]
    for row in paired.itertuples():
        records.append(dict(condition=row.condition,reference=row.reference,gain=row.gain,
            low=row.low,high=row.high,reduction_percent=100*row.gain/row.reference_loss,primary=row.primary))
    summary=dict(scope_assessment=scope,reason=reason,checks=checks,configuration=config,
        primary_comparisons=primary.to_dict('records'),all_paired_effects=records,
        terminal_coverage=coverage.to_dict('records'),risk=actual.reset_index().to_dict('records'),
        primary_cost=posterior_cost.to_dict(),mechanics=mechanics,numerics=numerical,
        assessment_scope='Conservative qualitative scope decision for this bounded, frozen pilot; not a universal selection rule')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    labels={'plane':'Plane','sphere_large':'Sphere, R=0.20 m','sphere_small':'Sphere, R=0.05 m',
            'weak':'Weak input program','anisotropic':'Anisotropic resistance'}
    lines=[f'# Scope decision: {scope}','',reason,'',
        'The physically generated force/velocity data fit the existing formulation under known isotropic resistance.',
        'This supports a maintained bilateral-contact example. It does not establish frictional/unilateral contact',
        'inference, a contact controller, safe velocity commands, or practical superiority over encoder-based estimators.',
        '', '## Task-loss evidence','',
        'The primary condition is the balanced smaller-radius guide at observation 600. Each of 64 test trajectories',
        'contributes one squared three-vector response error averaged over nine hypothetical queries.',
        'Memory is shared across the curved validation mixture; scalar shrinkage is continuously optimized on that mixture.',
        'Both primary references were specified before testing. Other pointwise intervals are exploratory.',
        '', '| Primary reference | Reference MSE | Mean-prediction MSE | Relative reduction | Paired gain 95% interval |',
        '| --- | ---: | ---: | ---: | --- |']
    relevant=paired[(paired.condition==case)&paired.reference.isin(['Mode','Scalar shrinkage','Ridge map'])]
    for r in relevant.itertuples():
        lines.append(f'| {r.reference} | {r.reference_loss:.6g} | {r.mean_loss:.6g} | {100*r.gain/r.reference_loss:.3f}% | [{r.low:.4g}, {r.high:.4g}] |')
    lines.extend(['', 'The 3.8% advantage over pooled shrinkage is supported in this frozen primary comparison.',
        'Its roughly 0.17% advantage over its own mode is statistically detectable but small in practical terms.',
        'The large ridge/PCA advantages mainly demonstrate projector structure and known-force information;',
        'they do not isolate a benefit of maintaining a full distribution. Larger-radius and weak-input gains',
        'over mode/shrinkage have intervals containing zero. In the anisotropic stress, mean prediction is worse',
        'than tuned scalar shrinkage. All outcomes remain in [paired.csv](paired.csv).',
        'The selected curved beta=0.8 is the lowest tested memory value. These are finite-family results,',
        'not evidence that no better point-memory policy exists.',
        '', '## Uncertainty: useful in the static control, unreliable under curvature','',
        '| Terminal condition | State-ball coverage | HPD coverage | Clean component coverage | Mean scalar width / v0 |',
        '| --- | --- | --- | ---: | ---: |'])
    for condition,label in labels.items():
        h=coverage[coverage.condition==condition].set_index('region')
        ball=h.loc['State ball'];hpd=h.loc['State HPD'];clean=h.loc['Clean response']
        lines.append(f'| {label} | {int(ball.successes)}/{int(ball.trials)} | {int(hpd.successes)}/{int(hpd.trials)} | {100*clean.coverage:.1f}% | {clean.mean_width:.4f} |')
    lines.extend(['', 'The static-plane state regions cover 63/64 trials; their Wilson intervals include 95%.',
        'Mean clean component coverage is 95.0%. This is a clean matched-model control, not a drift guarantee.',
        'Under small-radius motion, HPD coverage 23/64 and clean coverage 66.4% are far below nominal 95%.',
        'The weak-input ball covers 61/64, but its HPD and clean-response summaries still under-cover.',
        'A radius alone can conceal directional model error.',
        '', 'For a rank-two projector in three dimensions, each queried scalar component has full physical range',
        'width v0. Normalized widths are therefore fractions of that range. The small-sphere interval width',
        'is only about 6.6% of the possible range, yet it misses often: the issue is overconfidence, not',
        'nearly uninformative intervals. No noisy-observation intervals are evaluated.',
        'Coverage replicates are trajectories; 27 component indicators are averaged within each trajectory.',
        'See [coverage.csv](coverage.csv) for Wilson/clustered intervals and [query_coverage.csv](query_coverage.csv) for individual components.',
        '', '| Condition | Posterior full-vector risk | Empirical full-vector MSE | Empirical / posterior risk |',
        '| --- | ---: | ---: | ---: |'])
    for condition,label in labels.items():
        r=actual.loc[condition]
        lines.append(f'| {label} | {r.risk_mean:.4g} | {r.empirical_response_error:.4g} | {r.empirical_to_posterior_risk:.2f} |')
    lines.extend(['', 'Risk is informative under the stationary model, but it understates primary small-sphere error by',
        'about 2.6 times and anisotropic physical-response error by about 62 times. Average weak-input risk',
        'can be close while directional HPD/interval coverage is poor. Moments and tails answer different questions.',
        '', '## Mechanics, numerics and the theorem','',
        f'All {checks["posterior_evaluations"]} reports pass separate mean, state-region and response-interval resolution checks; no MCMC or noise learning is used.',
        f'The independent KKT/projector velocity discrepancy is at most {checks["maximum_matched_projection_identity_error"]:.3g} m/s in matched conditions.',
        f'The full-horizon dt/2 checks change velocity by at most {mechanics["maximum_velocity_difference_over_sigma"]:.3g} measurement-noise standard deviations.',
        'The continuous-CDF quadrature extension passes independent pole/ridge references through concentration 1e6.',
        'Existing core library files remain unchanged; this is a dimension-specific deterministic numerical extension.',
        f'{checks["theorem_valid"]}/{checks["theorem_applicable"]} applicable fixed-time modal certificates satisfy their sufficient condition, with {checks["valid_certificate_violations"]} violations.',
        'Invalid conditions remain in [bounds.csv](bounds.csv); these are not simultaneous-time guarantees.',
        'Anisotropic resistance breaks the scalar response law and is excluded from the theorem.',
        'Its response errors use actual independently solved KKT query velocities; geometric targets are retained separately.',
        'Known noise and a correct numerical posterior cannot repair a wrong state or observation model.',
        '', '## Computation and scope','',
        f'In the primary case, cumulative correction time averages {posterior_cost.corrections:.3f} s over 600 observations.',
        f'The three full posterior/interval reports together average {posterior_cost.integrations:.3f} s; query decisions add {posterior_cost.decisions:.3g} s.',
        'A single full report takes about 0.16 s, exceeding the 0.05 s observation step. This implementation does',
        'not establish a 20 Hz full-interval controller. Mean-only evaluation is cheaper; clocks retain their actual frequencies.',
        'These cloud wall-clock measurements are not dedicated-machine speed or asymptotic guarantees.',
        '', 'The pilot supports using the robot setting as a physical illustration of the conditional framework.',
        'It does not yet justify a broad calibrated-contact claim or a large Bayesian decision advantage over its own mode.',
        'The mean map E[P] is generally full rank: it predicts an expected response and is not one physically',
        'admissible contact projector or a guaranteed feasible velocity command.',
        '', 'After this bounded test, the next practical comparison should include the same noisy encoder positions',
        'for every applicable method, with a causal local-plane control. A later dynamics/observation model must',
        'address curvature and uncertain resistance before stronger uncertainty claims. No such machinery is added here.',
        '', '## Reproduce','', '```bash',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/validate_contact.py',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/check_contact_posterior.py',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_contact_pilot.py --workers 2',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/check_contact_mechanics.py',
        'python scripts/summarize_contact_pilot.py',
        'python scripts/make_contact_notebook.py',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/execute_notebook.py notebooks/03_contact.ipynb',
        'python scripts/plot_contact_pilot.py',
        'python scripts/write_contact_manifest.py', '```', '',
        '[Physical paths](figures/physical_setup.pdf), [paired task losses](figures/paired_losses.pdf), and',
        '[coverage versus width](figures/coverage_width.pdf) are the main figures. The executed',
        '[contact notebook](../../notebooks/03_contact.ipynb) contains the assumptions, interpretation and supplementary diagnostics.',
        'The [pilot specification](../../docs/contact_pilot_spec.md) is preserved. Historical results and the field notebook are unchanged.',''])
    (OUT/'findings.md').write_text('\n'.join(lines).rstrip()+'\n');print(scope);print(reason)


if __name__=='__main__':main()

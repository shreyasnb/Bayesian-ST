"""Rebuild the contact interpretation from trajectory-level saved evidence."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/contact_tracking'


def paired(x, y):
    """Positive gain means x has greater loss than y; all planned pairs required."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or not np.isfinite(x-y).all():
        raise ValueError('An unresolved or missing pair cannot be silently omitted')
    difference = x-y
    half = student_t.ppf(.975, len(x)-1)*difference.std(ddof=1)/np.sqrt(len(x))
    return dict(gain=float(difference.mean()), low=float(difference.mean()-half),
        high=float(difference.mean()+half), trials=len(x),
        relative_gain=float(difference.mean()/x.mean()))


def main():
    mean = pd.read_csv(OUT / 'mean/metrics.csv')
    frames = OUT / 'uncertainty'
    metrics = pd.read_csv(frames / 'metrics.csv')
    summary = pd.read_csv(frames / 'summary.csv')
    shape = pd.read_csv(frames / 'shape_comparison.csv')
    directions = pd.read_csv(frames / 'directional.csv')
    selected = json.loads((frames / 'selected.json').read_text())
    matched_path = ROOT / 'results/contact_uncertainty/matched/summary.csv'
    matched = pd.read_csv(matched_path) if matched_path.exists() else None
    end = metrics.query("condition == 'sphere_small' and t == 600").set_index(['method','trial']).sort_index()
    def rows(method): return end.loc[method].sort_index()
    def avg(method, field): return float(rows(method)[field].mean())
    original = mean.query("condition == 'sphere_small' and t == 600 and method == 'original' and noise == 'on'").sort_values('trial')
    motion = mean.query("condition == 'sphere_small' and t == 600 and method == 'velocity' and noise == 'on'").sort_values('trial')
    mean_gain = paired(original.corrected_mse, motion.corrected_mse)
    average_gain = paired(original.time_average_mse, motion.time_average_mse)
    process_gain = paired(rows('gaussian_motion').clean_mse, rows('process_motion').clean_mse)
    score_gain = paired(rows('gaussian_motion').interval_score, rows('process_motion').interval_score)
    process_time_gain = paired(rows('gaussian_motion').time_mean_mse, rows('process_motion').time_mean_mse)
    process_time_vs_raw = paired(rows('raw').time_mean_mse, rows('process_motion').time_mean_mse)
    point_gain = paired(rows('temperature').regularized_mse, rows('process_motion').clean_mse)
    temp = next(p['temperature'] for p in selected['selected'] if p['scope']=='curved' and p['method']=='temperature')
    main_methods = ['raw','temperature','process_identity','process_motion','gaussian_identity','gaussian_motion']
    labels = dict(raw='Bingham, selected memory/forecast',temperature='Bingham, calibrated spread',
        process_identity='Bingham process, identity motion',process_motion='Bingham process, causal motion',
        gaussian_identity='Gaussian process, identity motion',gaussian_motion='Gaussian process, causal motion')
    conditions = dict(plane='Stationary plane',sphere_large='20 cm sphere',
        sphere_small='5 cm sphere',weak='Forces mostly along one axis',
        anisotropic='Anisotropic resistance')
    table = ['| Prediction | MSE (mm/s)² | Time-average MSE (mm/s)² | Clean coverage | Width (mm/s) | Interval score (mm/s) |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for method in main_methods:
        table.append(f"| {labels[method]} | {avg(method,'clean_mse')*1e6:.4f} | {avg(method,'time_mean_mse')*1e6:.4f} | {avg(method,'clean_coverage'):.3f} | {avg(method,'clean_width')*1e3:.3f} | {avg(method,'interval_score')*1e3:.3f} |")
    process_ms = avg('process_motion','end_to_end_seconds')/600*1e3
    gaussian_ms = avg('gaussian_motion','end_to_end_seconds')/600*1e3
    report_times = metrics.query("condition == 'sphere_small' and method in ['process_motion','gaussian_motion']").groupby(['method','trial']).report_seconds.mean().groupby('method').mean()*1e3
    cost_scope = 'In-study wall time with concurrent workers'
    timing_path = OUT/'timing/summary.csv'
    if timing_path.exists():
        timing = pd.read_csv(timing_path).set_index('method')
        process_ms = float(timing.loc['process_motion','estimation_ms_per_observation_median'])
        gaussian_ms = float(timing.loc['gaussian_motion','estimation_ms_per_observation_median'])
        for method in ['process_motion','gaussian_motion']:
            report_times[method] = float(timing.loc[method,'three_report_posterior_seconds_median'])*1e3/3
        cost_scope = 'Median of three separate serial runs of preassigned small-sphere trajectory 0; shared forecast included'
    shape_small = shape.query("condition == 'sphere_small' and t == 600")
    family = []
    for method in ['temperature','process_identity','process_motion']:
        exact, local = rows(method), rows(method+'_laplace')
        delta = paired(local.clean_mse, exact.clean_mse)
        family.append(dict(method=method,**delta,
            exact_coverage=float(exact.clean_coverage.mean()),local_coverage=float(local.clean_coverage.mean()),
            exact_width=float(exact.clean_width.mean()),local_width=float(local.clean_width.mean()),
            exact_failures=int((~exact.numerical_ok.astype(bool)).sum()),
            local_failures=int((~local.numerical_ok.astype(bool)).sum())))
    evidence = dict(terminal_original_mode_mse=float(original.corrected_mse.mean()),
        terminal_causal_mode_mse=float(motion.corrected_mse.mean()),causal_vs_original=mean_gain,
        causal_vs_original_time_average=average_gain,
        process_vs_gaussian=process_gain,process_vs_gaussian_interval_score=score_gain,
        process_vs_gaussian_time_average=process_time_gain,
        process_vs_raw_time_average=process_time_vs_raw,
        process_vs_regularized_point=point_gain,
        temperature=temp,process_ms_per_observation=process_ms,gaussian_ms_per_observation=gaussian_ms,
        process_ms_per_report=float(report_times['process_motion']),gaussian_ms_per_report=float(report_times['gaussian_motion']),
        shape=family,cost_scope=cost_scope,source='Independent trajectory rows; no queries or components counted as replicates')
    (OUT/'interpretation.json').write_text(json.dumps(evidence,indent=2)+'\n')
    def gain_text(g): return f"{g['gain']*1e6:.4g} [{g['low']*1e6:.4g}, {g['high']*1e6:.4g}] (mm/s)²"
    process_answer = ('has lower terminal MSE' if process_gain['low']>0 else
        'has higher terminal MSE' if process_gain['high']<0 else 'has no resolved terminal MSE advantage')
    axis = summary[(summary.condition=='weak') & summary.method.isin(
        ['temperature','process_motion','gaussian_motion'])].set_index('method')
    compact = f'''The comparison is paired across **64 independent small-sphere trajectories**. Positive paired gains below mean lower loss for the Bingham process prediction; intervals are pointwise 95% Student intervals.

| Question | Held-out result |
| --- | --- |
| Does the causal forecast reduce lag? | Terminal mode MSE changes from {original.corrected_mse.mean()*1e6:.3f} to {motion.corrected_mse.mean()*1e6:.3f} (mm/s)², a {100*mean_gain['relative_gain']:.1f}% reduction. Time-average MSE improves {100*average_gain['relative_gain']:.1f}%. These gains are available to point methods. |
| What does spread calibration change? | With T={temp:g}, clean coverage changes from {avg('raw','clean_coverage'):.1%} to {avg('temperature','clean_coverage'):.1%}; width from {avg('raw','clean_width')*1e3:.3f} to {avg('temperature','clean_width')*1e3:.3f} mm/s; interval score from {avg('raw','interval_score')*1e3:.3f} to {avg('temperature','interval_score')*1e3:.3f} mm/s (lower is better). The mode is unchanged. |
| Does process Bingham beat the cheap Gaussian? | It {process_answer}. The Gaussian-minus-Bingham paired MSE difference is {gain_text(process_gain)} ({100*process_gain['relative_gain']:.2f}% of Gaussian loss). |
| What does computation cost? | Measured filter cost, including the common forecast tracker, is {process_ms:.2f} versus {gaussian_ms:.2f} ms/observation. One 27-component uncertainty report averages {report_times['process_motion']:.0f} versus {report_times['gaussian_motion']:.0f} ms. |

For uncertainty, Bingham/Gaussian clean-component coverage is **{avg('process_motion','clean_coverage'):.1%}/{avg('gaussian_motion','clean_coverage'):.1%}**, with mean interval width **{avg('process_motion','clean_width')*1e3:.3f}/{avg('gaussian_motion','clean_width')*1e3:.3f} mm/s** and interval score **{avg('process_motion','interval_score')*1e3:.3f}/{avg('gaussian_motion','interval_score')*1e3:.3f} mm/s** (lower is better). The paired Gaussian-minus-Bingham score difference is {score_gain['gain']*1e3:.4f} [{score_gain['low']*1e3:.4f}, {score_gain['high']*1e3:.4f}] mm/s. Lower MSE alone would not establish better uncertainty.

The whole-trajectory tradeoff differs from the terminal result: time-average MSE is {avg('raw','time_mean_mse')*1e6:.3f} for the raw forecast, {avg('process_motion','time_mean_mse')*1e6:.3f} for process Bingham and {avg('gaussian_motion','time_mean_mse')*1e6:.3f} (mm/s)² for the Gaussian. Point memory was selected for terminal MSE, so this is a secondary comparison of the retained policies, not a search for the best possible time-average point estimator. Temperature was fitted on the curved-condition validation mixture; more coverage on this one endpoint can mean unnecessarily wide intervals.

An exploratory case shows where process uncertainty is useful: when the second and third force components are divided by five, the temperature control covers {axis.loc['temperature','clean_coverage']:.1%} at width {axis.loc['temperature','clean_width']*1e3:.2f} mm/s. Process Bingham gives {axis.loc['process_motion','clean_coverage']:.1%} at {axis.loc['process_motion','clean_width']*1e3:.2f} mm/s; the Gaussian gives {axis.loc['gaussian_motion','clean_coverage']:.1%} at {axis.loc['gaussian_motion','clean_width']*1e3:.2f} mm/s. Useful uncertainty propagation is supported there, while the full Bingham family still has no demonstrated premium over the Gaussian.

Time-average loss, same-parameter Gaussian checks, all misses and the other contact conditions are in the [full findings](../results/contact_tracking/findings.md). Cost scope: {cost_scope.lower()}. Average report cost does not guarantee a 50 ms deadline. These numbers separate improved dynamics from any additional benefit of a distribution. The finite validation scan selected a lag of 32 at the extended boundary; it does not establish a globally optimal forecast.
'''
    (OUT/'notebook_interpretation.md').write_text(compact)
    lines = ['# Contact tracking findings','',
        'The study preserves the earlier contact pilot and uses fresh roots 6245201/6245202: 16 validation and 64 test trajectories per condition, with nine queries averaged within each trajectory. The primary task is clean vector-response MSE, with the terminal small-sphere comparison predeclared for inference. Time-average MSE and clean interval score are retained secondary comparisons. One curved policy is shared across both radii and the one-axis force program. The plane is selected separately; anisotropic resistance never tunes.','',
        compact.replace('[full findings](../results/contact_tracking/findings.md)','tables below'),'','## Small-sphere endpoint','',*table,'',
        f"The regularized point reference has terminal MSE {avg('temperature','regularized_mse')*1e6:.4f} (mm/s)². Its scale rho is optimized continuously on validation for the improved practical forecast. Its paired loss minus Bingham process loss is {gain_text(point_gain)}. A positive comparison against this point alone is not evidence of superiority over the Gaussian uncertainty control.",'',
        '## Mean tracking and causality','',
        f"Original mode minus causal mode terminal MSE is {gain_text(mean_gain)}; the time-average difference is {gain_text(average_gain)}. Predictions use observations through t-1; the current observation is read only for correction. The mode remains equal to transported weighted projector least squares. Noiseless runs keep the same mechanics, forces, variance and memory; truth alignment is a diagnostic, never a practical tuning candidate.",'',
        'The selected finite-family causal forecast uses beta=0.9, lag=32 and damping=1 on the curved conditions. Lag 32 won the single permitted extension from 16 and remains a boundary. All finite-grid conclusions are conditional on that search. Shorter exponential memory did not replace beta=0.8 as its family winner.','',
        '## Distribution shape','',
        '| Same-parameter comparison | Local minus full MSE, 95% paired interval (mm/s)² | Full/local clean coverage | Full/local width (mm/s) | Unresolved full/local |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in family:
        lines.append(f"| {labels[r['method']]} | {gain_text(r)} | {r['exact_coverage']:.3f}/{r['local_coverage']:.3f} | {r['exact_width']*1e3:.3f}/{r['local_width']*1e3:.3f} | {r['exact_failures']}/{r['local_failures']} |")
    lines += ['', '## State regions','',
        '| Method | Chordal-ball inclusion | HPD or local-ellipse inclusion | Region type |',
        '| --- | ---: | ---: | --- |']
    for method in main_methods:
        g = rows(method)
        region = 'Local Gaussian ellipse' if method.startswith('gaussian') else 'Invariant-density Bingham HPD'
        lines.append(f"| {labels[method]} | {int(g.ball_covered.sum())}/64 | {int(g.hpd_covered.sum())}/64 | {region} |")
    lines += ['',
        'These are terminal state inclusions, not componentwise response coverage. Counts use the same 64 independent trajectories; they are not simultaneous-time guarantees. A Gaussian local ellipsoid is not an invariant-density Grassmann HPD region.']
    lines += ['', '## Mean, mode and regularized point predictions','',
        '| Filter | Mode MSE (mm/s)² | Mean MSE (mm/s)² | Regularized mode MSE (mm/s)² |',
        '| --- | ---: | ---: | ---: |']
    for method in main_methods:
        lines.append(f"| {labels[method]} | {avg(method,'mode_mse')*1e6:.5f} | {avg(method,'clean_mse')*1e6:.5f} | {avg(method,'regularized_mse')*1e6:.5f} |")
    lines += ['',
        'Every row uses that filter\'s own forecast and process policy. Regularization predicts rho*P_mode*w, with rho fitted continuously on pooled terminal validation MSE. It is not a coarse scalar grid. The posterior mean is optimal for squared loss under its own maintained distribution; it need not dominate its mode on every fixed physical path.']
    lines += ['', '## Posterior risk and directional error','',
        '| Method | Posterior vector risk (mm/s)² | Actual vector MSE (mm/s)² | Actual / predicted |',
        '| --- | ---: | ---: | ---: |']
    for method in ['temperature','process_motion','gaussian_motion']:
        risk_value = avg(method,'posterior_risk')
        lines.append(f"| {labels[method]} | {risk_value*1e6:.4f} | {avg(method,'clean_mse')*1e6:.4f} | {avg(method,'clean_mse')/risk_value:.3f} |")
    lines += ['',
        'The risk is the within-posterior full-vector quantity w^T(M-M^2)w, averaged over the nine queries. Actual error uses the same full-vector loss, not a single-coordinate variance. A difference measures descriptive risk calibration in this physical experiment; equality is not guaranteed under the misspecified dynamics.','',
        '| Method | Physical direction | Signed bias (deg) | RMS error (deg) | RMS posterior SD (deg) |',
        '| --- | --- | ---: | ---: | ---: |']
    directional_end = directions.query("condition == 'sphere_small' and t == 600")
    for (method,direction), group in directional_end.groupby(['method','direction'],sort=False):
        if method not in ['temperature','process_motion','gaussian_motion']:
            continue
        lines.append(f"| {labels[method]} | {direction} | {np.degrees(group.tangent_error.mean()):.3f} | {np.degrees(np.sqrt(np.mean(group.tangent_error**2))):.3f} | {np.degrees(np.sqrt(np.mean(group.posterior_sd**2))):.3f} |")
    lines += ['',
        'Directions are the actual mechanical normal-motion direction projected at the estimated normal and its perpendicular. Geometry enters scoring only. Antipodal signs are aligned in scoring so arbitrary eigensolver signs do not cancel bias. Bingham widths use shortest projective logarithms; Gaussian covariance is a local-coordinate approximation. These standard deviations are not central credible intervals or an HPD claim.']
    lines += ['',
        'The same-parameter comparison isolates reporting shape. The recursive Gaussian additionally uses a local transition and a linearized observation update. Its normalized predictive log score is the linearized Gaussian score after a six-observation, data-only Bingham warm-up. Both methods pay for that initialization and for the common causal-forecast tracker. The Gaussian local ellipsoid is not an invariant-density Grassmann HPD region.','',
        f'The secondary time-average raw-minus-process gain is {gain_text(process_time_vs_raw)}, while Gaussian-minus-Bingham time-average gain is {gain_text(process_time_gain)}. Memory and regularization were fitted to terminal point loss and process spread to predictive NLL. This comparison does not optimize all methods for the time-average endpoint. Tiny same-parameter reporting differences are descriptive and are not a certified tail-accuracy advantage.','',
        '## Other conditions and numerical flags','',
        '| Condition | Method | MSE (mm/s)² | Clean coverage | Width (mm/s) | Unresolved reports |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for r in summary[summary.method.isin(['temperature','process_motion','gaussian_motion'])].itertuples():
        lines.append(f'| {conditions.get(r.condition,r.condition)} | {labels[r.method]} | {r.clean_mse*1e6:.4f} | {r.clean_coverage:.3f} | {r.clean_width*1e3:.3f} | {r.unresolved} |')
    lines += ['',
        'For forces mostly along one axis, the second and third force components are divided by five. Local Fisher information and actual retained-history turning are recorded independently. Anisotropic resistance breaks the assumed mobility map: its truth is the KKT response, not an isotropic projector surrogate. In the Stage 1 mode-only ablation, removing sensor noise does not remove its large response-prediction error; Stage 2 does not evaluate noiseless coverage.','',
        'Numerical flags concern mean integration, region integration, response quantiles and local chart validity separately. A fitted predictive moment does not certify density tails. The software gate can pass while empirical coverage fails; the gate does not test whether the physical simulation obeys a fitted stochastic transition.','',
        '## Cost and scope','',
        f'The Bingham/Gaussian process filters cost {process_ms:.3f}/{gaussian_ms:.3f} ms per observation including their shared forecast tracker, with uncertainty reports averaging {report_times["process_motion"]:.1f}/{report_times["gaussian_motion"]:.1f} ms each for nine queries and three components. Cost scope: {cost_scope}. Component clocks are nested: inverse-fit time is inside prediction, and warm-up inside the Gaussian path. The end-to-end wall clock avoids adding nested clocks twice. Contended in-study timings remain in the original tables. These are implementation measurements, not hardware-independent rates or worst-case latency guarantees.', '',
        'Cheap point and local Gaussian methods should remain the default accuracy/cost references. A full Bingham distribution needs a demonstrated advantage in decision loss, calibrated width/score, or broad global geometry to justify additional computation; a gain from using known force, enforcing a projector, or predicting motion does not establish that advantage. Posterior means are expected-response predictions, not contact controllers.','',
        'For these held-out contacts, the primary MSE and interval-score intervals do not establish a Bingham-process advantage over the recursive Gaussian. The raw motion predictor is better at the terminal endpoint; process propagation helps the retained policies over the full trajectory. This supports a motion/identification example, not a claim that full-distribution computation is necessary for this robotics task. The optional uncertainty-based abstention decision was not triggered by a material demonstrated premium over the cheap control.','',
        'The identity nominal frame does not make the deterministic modal theory vacuous. That bound pays for actual residual drift and can lose its sufficient condition. Credible-region calibration on arbitrary drifting paths is a different statement and is not implied by that bound. Learned G, temperature and assumed-density projection require separate assumptions and empirical or theoretical justification.','',
        'No manuscript is changed. A physical seven-joint extension is deferred to the bounded numerical feasibility check: increasing ambient dimension cannot repair unmodelled dynamics, and known kinematic constraints must be supplied to the structured competitor.']
    (OUT/'findings.md').write_text('\n'.join(lines)+'\n')
    uncertainty = ['# Process uncertainty and distribution-family comparison','',
        'The final physical comparison follows the later contact tracking guide. It first selects a causal mean forecast, then separately selects report temperature and process spread. It replaces the earlier proposed broad beta/likelihood-scale scan with the requested bounded Stage 1/Stage 2 design. The raw historical pilot remains unchanged.','',compact.replace('../results/contact_tracking/findings.md','../contact_tracking/findings.md'),'',
        'Full physical results and limitations: [contact tracking findings](../contact_tracking/findings.md).', '',
        '## Matched process','']
    if matched is not None:
        uncertainty += ['| Kappa | Method | Trials | MSE | Clean coverage | Interval score | Unresolved |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
        uncertainty += [f'| {r.kappa:g} | {r.method} | {r.trials} | {r.clean_mse:.5g} | {r.clean_coverage:.3f} | {r.interval_score:.4f} | {r.numerical_failures} |' for r in matched.itertuples()]
        uncertainty += ['',
            'This dimensionless process has known kappa and uniform initialization. It uses 200 independent trajectories per kappa; the first 16 per kappa are the full-density reference subset, fixed before testing. Do not compare a 16-trajectory reference rate directly with a 200-trajectory method rate as if they were paired. The complete matched metrics retain method rows on that same subset for paired inspection. Refined full-density failures remain visible; successful development is not a certificate for every held-out target.']
    uncertainty += ['',
        '## What the experiment can support','',
        '1. Explicit propagation is judged after mean and spread calibration, using proper scores and widths as well as coverage. A dynamics improvement is available to the point and Gaussian controls.','',
        '2. The same-A Laplace comparison isolates global density shape; the recursive Gaussian comparison also includes local transition and measurement approximations. Broad, ambiguous posteriors and concentrated physical paths are reported separately.','',
        '3. The physical plant is not the isotropic Watson transition. Good matched-process calibration does not establish physical robustness; a wrong resistance law remains a likelihood failure.','',
        '4. Correction and mean prediction are distinguished from occasional 27-component continuous-CDF reports. Use measured wall time and flags, not only the conjugate update cost, to judge a 20 Hz implementation.','',
        '5. An uncertainty-led paper needs a material width/score/decision advantage beyond the cheap Gaussian. Where that evidence is absent or numerically unresolved, the defensible claim is narrower identification, exact conditional updating and controlled theory/numerics, not general calibrated contact prediction.']
    (ROOT/'results/contact_uncertainty/findings.md').write_text('\n'.join(uncertainty)+'\n')
    print(json.dumps(evidence,indent=2))


if __name__ == '__main__':
    main()

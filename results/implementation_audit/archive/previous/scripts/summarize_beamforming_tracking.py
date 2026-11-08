"""Paired trajectory analyses of frozen passive tracking/process outputs.

No numerical-success filtering is used. Missing endpoints or unresolved flags
remain visible. A full CP1 reference comparison uses its predeclared first
trajectory subset, not the larger native mean-only cohort.
"""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from scipy.stats import t as student_t
from lib.beamforming_validation import verify_component


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired(group,method,reference,endpoint,count,flag='mean_ok'):
    identities=np.arange(count);selected=group[group.method.isin([method,reference])]
    wide=selected.pivot(index='trial',columns='method',values=endpoint).reindex(identities)
    flags=selected.pivot(index='trial',columns='method',values=flag).reindex(identities)
    if method not in wide or reference not in wide:
        difference=np.full(count,np.nan);resolved=np.zeros(count,bool)
    else:
        difference=(pd.to_numeric(wide[reference],errors='coerce')-pd.to_numeric(wide[method],errors='coerce')).to_numpy()
        resolved=(flags[method].eq(True)&flags[reference].eq(True)).to_numpy(bool)
    finite=np.isfinite(difference);complete=bool(finite.all())
    mean=float(difference.mean()) if complete else np.nan
    half=float(student_t.ppf(.975,count-1)*difference.std(ddof=1)/np.sqrt(count)) if complete and count>1 else np.nan
    return dict(method=method,reference=reference,endpoint=endpoint,trials=count,
        finite_pairs=int(finite.sum()),resolved_pairs=int(resolved.sum()),
        gain=mean,lower=mean-half,upper=mean+half,
        numerical_status='resolved' if complete and resolved.all() else 'unresolved or missing endpoint',
        interpretation='Positive favors method for loss/power/width. Coverage alone is not a proper score.',
        inference='Exploratory pointwise paired Student interval across independent trajectories')


def risk_bins(frame,keys):
    rows=[]
    for setting,g in frame.groupby(keys,sort=False,dropna=False):
        setting=setting if isinstance(setting,tuple) else (setting,)
        planned_count=len(g)
        g=g[np.isfinite(g.projection_risk)&np.isfinite(g.projection_mse)].sort_values(['projection_risk','trial'])
        omitted=planned_count-len(g)
        for number,positions in enumerate(np.array_split(np.arange(len(g)),5),1):
            if not len(positions):continue
            h=g.iloc[positions];loss=h.projection_mse.to_numpy();n=len(h)
            half=float(student_t.ppf(.975,n-1)*loss.std(ddof=1)/np.sqrt(n)) if n>1 else np.nan
            rows.append(dict(zip(keys,setting),bin=number,independent_trajectories=n,
                mean_resolved_trajectories=int(h.mean_ok.eq(True).sum()),
                minimum_predicted_risk=float(h.projection_risk.min()),maximum_predicted_risk=float(h.projection_risk.max()),
                predicted_risk=float(h.projection_risk.mean()),actual_loss=float(loss.mean()),
                actual_low=float(loss.mean()-half),actual_high=float(loss.mean()+half),
                omitted_nonfinite=omitted,
                note='Queries were averaged within each trajectory before sorting into five forecast-risk bins.'))
    return pd.DataFrame(rows)


def analyze(name,component):
    document=verify_component(ROOT,name,component);folder=ROOT/'results/beamforming_unified'/name/component
    config=document['config'];metrics=pd.read_csv(folder/'metrics.csv');rows=[]
    if component=='tracking':
        keys=['s','t','policy']
        for setting,g in metrics.groupby(keys,sort=False):
            for reference in ['Mode','Matched isotropization','Oracle']:
                for endpoint in ['population_power','heldout_power','projection_mse']:
                    rows.append(dict(zip(keys,setting),**paired(g,'Posterior mean',reference,endpoint,config['test_trials'])))
        risk=risk_bins(metrics,['s','t','policy','method'])
        regions=pd.read_csv(folder/'coverage.csv');region_rows=[]
        for (state,t),g in regions.groupby(['state','t'],sort=False):
            for field in ['ball_covered','hpd_covered']:
                n=len(g);success=int(g[field].sum());p=success/n;z=1.959963984540054
                middle=(p+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
                region_rows.append(dict(state=state,t=t,region=field,trajectories=n,covered=success,coverage=p,
                    lower=middle-half,upper=middle+half,state_mass_failures=int((~g.state_ok.astype(bool)).sum()),
                    separate_scalar_tail_failures=int((~g.tail_ok.astype(bool)).sum()),
                    scope='Known powers, fixed state, beta=1, uniform ambient-line prior; state inclusion only'))
        pd.DataFrame(region_rows).to_csv(folder/'state_coverage_summary.csv',index=False)
        additional=['state_coverage_summary.csv']
    else:
        keys=['n','kappa','t']
        for setting,g in metrics.groupby(keys,sort=False):
            for reference in ['Power forgetting','Identity prediction','Full-density CP1 reference']:
                if reference not in set(g.method):continue
                for endpoint in ['population_power','projector_mean_mse','projection_mse','interval_score','clean_width','clean_coverage']:
                    scalar=endpoint in ['interval_score','clean_width','clean_coverage']
                    count=(config['reference_trials'] if reference=='Full-density CP1 reference' else config['process_trials'])
                    if scalar:count=min(count,config['process_tail_trials'])
                    row=paired(g,'Process Bingham',reference,endpoint,count,'tail_ok' if scalar else 'mean_ok')
                    row['subset']='Predeclared first tail/reference trajectories' if count<config['process_trials'] else 'All predeclared native trajectories'
                    rows.append(dict(zip(keys,setting),**row))
        risk=risk_bins(metrics,['n','kappa','t','method']);additional=[]
        scores=pd.read_csv(folder/'scores.csv');prospective=[]
        reference_checks=pd.read_csv(folder/'reference_checks.csv')
        reference_guard=reference_checks.groupby(['kappa','trial']).reference_numerical_ok.all().to_dict()
        for setting,g in scores.groupby(['n','kappa','trial','method'],sort=False):
            values=pd.to_numeric(g.log_score,errors='coerce').to_numpy()
            finite=len(values)==config['process_T'] and np.isfinite(values).all()
            resolved=finite and (setting[-1]!='Full-density CP1 reference' or reference_guard.get((setting[1],setting[2]),False))
            prospective.append(dict(zip(['n','kappa','trial','method'],setting),
                negative_log_score=float(-values.mean()) if finite else np.nan,mean_ok=resolved,
                observations=len(values),finite_scores=int(np.isfinite(values).sum())))
        prospective=pd.DataFrame(prospective);score_pairs=[]
        for setting,g in prospective.groupby(['n','kappa'],sort=False):
            for reference in ['Power forgetting','Identity prediction','Full-density CP1 reference']:
                if reference not in set(g.method):continue
                count=config['reference_trials'] if reference=='Full-density CP1 reference' else config['process_trials']
                score_pairs.append(dict(zip(['n','kappa'],setting),
                    **paired(g,'Process Bingham',reference,'negative_log_score',count),
                    averaging='Prospective normalized log scores averaged over all times within each trajectory'))
        prospective.to_csv(folder/'predictive_by_trajectory.csv',index=False)
        pd.DataFrame(score_pairs).to_csv(folder/'predictive_paired.csv',index=False)
        additional.extend(['predictive_by_trajectory.csv','predictive_paired.csv'])
    pd.DataFrame(rows).to_csv(folder/'paired.csv',index=False);risk.to_csv(folder/'risk_calibration.csv',index=False)
    names=['paired.csv','risk_calibration.csv',*additional]
    metadata=dict(script_sha256=sha(__file__),experiment_source_hash=document['source_hash'],
        inputs={p:sha(folder/p) for p in ['metrics.csv','preregistered.json',*(['scores.csv'] if component=='process' else [])]},
        outputs={p:sha(folder/p) for p in names},
        analysis='No omission of unresolved pairs; finite descriptive estimates retain separate numerical-resolution status. Mean and scalar flags are endpoint-specific; the full reference-density guard remains visible in reference_checks.csv and is also propagated to full-reference trajectory log-score comparisons.')
    # The experiment manifest is extended only with postprocessing artifacts;
    # the filtering configuration and original source/table hashes stay intact.
    (folder/'paired_analysis.json').write_text(json.dumps(metadata,indent=2)+'\n')
    document['tables'].update({p:sha(folder/p) for p in [*names,'paired_analysis.json']})
    (folder/'manifest.json').write_text(json.dumps(document,indent=2)+'\n')
    print(name,component,'paired rows',len(rows),'risk bins',len(risk))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['quick','paper'],default='paper')
    parser.add_argument('--component',choices=['all','tracking','process'],default='all')
    args=parser.parse_args()
    for component in ['tracking','process'] if args.component=='all' else [args.component]:analyze(args.profile,component)


if __name__=='__main__':main()

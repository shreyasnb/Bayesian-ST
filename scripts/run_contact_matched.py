"""Matched Gr(2,3) process validation, with a fixed full-density subset.

The process is dimensionless and is not the mechanical sphere law. Independent
development seeds select the fixed reference resolution before test streams.
All targets, including failed local approximations and unresolved references,
are retained. The first sixteen trajectories per kappa are the reference subset.
"""
from pathlib import Path
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import hashlib,json,platform,sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,scipy
from lib.contact_spherical_reference import (SphericalReference,sample_watson,
    compare_reference,grid_basis,query_grid,grid_cdf)
from lib.contact_study import query_vectors
from lib.geometry import center
from lib.models import increment


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def portable(value):
    if isinstance(value,dict):return {str(k):portable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [portable(v) for v in value]
    if isinstance(value,np.generic):return portable(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def save_json(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(portable(value),indent=2,allow_nan=False));temporary.replace(path)


def observations(seed,trial,kappa,T=24,s=.12):
    rng=np.random.default_rng(np.random.SeedSequence([seed,int(kappa),trial,9027]))
    n=rng.normal(size=3);n/=np.linalg.norm(n);normals=[];U=[];V=[]
    for t in range(T):
        n=sample_watson(n,kappa,rng);v=np.eye(3)[:,t%3]
        U.append(v-n*(n@v)+np.sqrt(s)*rng.normal(size=3));V.append(v);normals.append(n.copy())
    return np.asarray(normals),np.asarray(U),np.asarray(V)


def development_check(out,config):
    directory=out/'reference';directory.mkdir(parents=True,exist_ok=True)
    declaration=dict(seed=config['development_seed'],trajectories_per_kappa=4,
        kappas=config['kappas'],T=config['T'],coarse_degree=24,fine_degree=32,
        coarse_summary_order=128,fine_summary_order=192,
        thresholds=dict(mean_error=2e-4,density_tv=.002,region_mass_error=.005,
                        scalar_cdf_error=.005,quantile_width_relative_error=.02),
        selection='Fixed development trajectories; no final test observation used')
    save_json(directory/'development_config.json',declaration)
    rows=[]
    for kappa in config['kappas']:
        for trial in range(4):
            normals,U,V=observations(config['development_seed'],trial,kappa)
            low=SphericalReference(kappa,24);high=SphericalReference(kappa,32)
            for j,(u,v) in enumerate(zip(U,V),1):
                low.update(u,v,config['s']);high.update(u,v,config['s'])
                if j not in config['report_times']:continue
                _,audit=compare_reference(low,high,query_vectors(1.),normals[j-1])
                rows.append(dict(kappa=kappa,trial=trial,t=j,**audit))
            print('Reference development',kappa,trial,'done',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(directory/'development_checks.csv',index=False)
    checks=dict(status='passed' if frame.reference_numerical_ok.all() else 'unresolved',
        targets=len(frame),failures=int((~frame.reference_numerical_ok).sum()),
        maxima={name:float(frame[name].max()) for name in ['density_tv','mean_error','region_mass_error',
                 'scalar_cdf_error','quantile_max_width_relative_error']},
        coarse_degree=24,fine_degree=32,coarse_summary_order=128,fine_summary_order=192,
        tables={'development_checks.csv':sha(directory/'development_checks.csv')},
        sources={name:sha(ROOT/name) for name in ['lib/contact_spherical_reference.py','scripts/run_contact_matched.py']},
        scope='Empirical harmonic/sphere-grid refinement; not a rigorous error enclosure')
    save_json(directory/'checks.json',checks)
    return checks


def interval_score(lower,upper,truth,alpha=.05):
    return (upper-lower)+(2/alpha)*(np.maximum(lower-truth,0)+np.maximum(truth-upper,0))


def record_summary(summary,kappa,trial,t,method,normal,W,pre_score,times=None):
    P=np.eye(3)-np.outer(normal,normal);truth=W@P
    lo=np.asarray(summary['response_lower']);hi=np.asarray(summary['response_upper']);M=np.asarray(summary['mean'])
    predicted=W@M.T;diagnostics=summary['numerics'];width=hi-lo
    covered=(truth>=lo)&(truth<=hi)
    row=dict(kappa=kappa,trial=trial,t=t,method=method,
        interval_score=float(np.mean(interval_score(lo,hi,truth))),clean_coverage=float(covered.mean()),
        clean_width=float(width.mean()),clean_mse=float(np.mean(np.sum((predicted-truth)**2,axis=1))),
        posterior_risk=float(np.mean(summary['risks'])),ball_covered=bool(summary.get('ball_covered',False)),
        hpd_covered=bool(summary.get('hpd_covered',False)),ball_radius=float(summary['ball_radius']),
        hpd_threshold=float(summary['hpd_threshold']),prior_log_score=pre_score,
        numerical_ok=bool(diagnostics.get('numerical_ok',False)),
        mean_numerical_ok=bool(diagnostics.get('mean_numerical_ok',diagnostics.get('numerical_ok',False))),
        region_numerical_ok=bool(diagnostics.get('regions_numerical_ok',diagnostics.get('numerical_ok',False))),
        interval_numerical_ok=bool(diagnostics.get('response_quantiles_ok',diagnostics.get('numerical_ok',False))),
        local_approximation_ok=bool(diagnostics.get('local_approximation_ok',True)),
        summary_seconds=float(diagnostics.get('integration_seconds',diagnostics.get('summary_seconds',np.nan))),
        cut_locus_mass=float(diagnostics.get('cut_locus_mass',0.)),
        region_kind=diagnostics.get('hpd_kind',diagnostics.get('region_kind','Bingham density-loss region')),
        unresolved_reason=diagnostics.get('unresolved_reason',''))
    if times:row.update(times)
    query_rows=[]
    for j in range(len(W)):
        for c in range(3):
            query_rows.append(dict(kappa=kappa,trial=trial,t=t,method=method,query=j,component=c,
                truth=float(truth[j,c]),mean=float(predicted[j,c]),lower=float(lo[j,c]),upper=float(hi[j,c]),
                covered=bool(covered[j,c]),numerical_ok=row['numerical_ok']))
    return row,query_rows


def trial_job(task):
    config,kappa,trial=task
    from lib.contact_process_prediction import predict_contact,fast_moments,predictive_log_score
    from lib.contact_gaussian import RecursiveTangentGaussian,laplace_summary
    from lib.contact_posterior import contact_summary
    normals,U,V=observations(config['test_seed'],trial,kappa,config['T'],config['s'])
    W=query_vectors(1.);A=np.zeros((3,3));D=np.zeros((3,3));M=2*np.eye(3)/3
    gaussian=RecursiveTangentGaussian(config['s'],kappa,warmup=config['warmup'])
    use_reference=trial<config['reference_trials']
    low=SphericalReference(kappa,24) if use_reference else None
    high=SphericalReference(kappa,32) if use_reference else None
    rows=[];queries=[];audits=[];shapes=[];scores=[]
    prediction_time=update_time=discount_time=0.;maximum_inverse_residual=0.;inverse_failures=0
    for t,(u,v) in enumerate(zip(U,V),1):
        B=increment(u,config['s'],v=v)
        start=perf_counter()
        predicted=predict_contact(A,kappa,previous_mean=M);prior=predicted.prior
        prediction_time+=perf_counter()-start
        maximum_inverse_residual=max(maximum_inverse_residual,predicted.fit.residual)
        inverse_failures+=int(not predicted.fit.converged)
        score=predictive_log_score(prior,u,v,config['s'])
        start=perf_counter();A=center(prior+B);evaluated=fast_moments(A);M=evaluated.mean
        update_time+=perf_counter()-start
        start=perf_counter();discount_prior=center(config['diagnostic_beta']*D)
        discount_score=predictive_log_score(discount_prior,u,v,config['s']);D=center(discount_prior+B)
        discount_time+=perf_counter()-start
        gaussian_score=gaussian.predictive_log_score(u,v);gaussian.update(u,v)
        reference_score=np.nan
        if use_reference:
            low.update(u,v,config['s']);reference_score=high.update(u,v,config['s'])
        scores.append(dict(kappa=kappa,trial=trial,t=t,process_bingham=score,
            recursive_gaussian=gaussian_score,discount_diagnostic=discount_score,reference=reference_score,
            gaussian_initialized=gaussian.initialized))
        if t not in config['report_times']:continue
        n=normals[t-1]
        summaries=[('Process Bingham',contact_summary(A,W,true_normal=n),score,
                    dict(prediction_seconds=prediction_time,update_seconds=update_time)),
                   ('Same-parameter Laplace',laplace_summary(A,W,true_normal=n),score,
                    dict(prediction_seconds=prediction_time,update_seconds=update_time,
                         score_kind='Underlying process Bingham prior; Laplace is reporting only')),
                   ('Recursive Gaussian',gaussian.summary(W,true_normal=n),gaussian_score,
                    {key:float(value) for key,value in gaussian.timings.items()}),
                   ('Fixed forgetting diagnostic',contact_summary(D,W,true_normal=n),discount_score,
                    dict(prediction_seconds=0.,update_seconds=discount_time))]
        for method,summary,pre,times in summaries:
            row,q=record_summary(summary,kappa,trial,t,method,n,W,pre,times)
            row.update(maximum_inverse_residual=maximum_inverse_residual,inverse_failures=inverse_failures)
            rows.append(row)
            if use_reference:queries.extend(q)
        if use_reference:
            reference,audit=compare_reference(low,high,W,n)
            audits.append(dict(kappa=kappa,trial=trial,t=t,**audit))
            reference['numerics']['numerical_ok'] &= audit['reference_numerical_ok']
            reference['numerics'].update(mean_numerical_ok=audit['mean_error']<2e-4,
                regions_numerical_ok=audit['region_mass_error']<.005,
                response_quantiles_ok=audit['scalar_cdf_error']<.005 and audit['quantile_max_width_relative_error']<.02)
            row,q=record_summary(reference,kappa,trial,t,'Full-density reference',n,W,reference_score,
                dict(prediction_seconds=high.prediction_seconds,update_seconds=high.update_seconds))
            rows.append(row);queries.extend(q)
            nodes,w,Y,_=grid_basis(high.degree,192,384)
            reference_density=high._density_from_basis(nodes,Y);reference_density=np.maximum(reference_density,0.)
            reference_density/=w@reference_density
            process_density=np.exp(np.trace(A)-np.einsum('ni,ij,nj->n',nodes,A,nodes)-evaluated.log_normalizer)
            process_density/=w@process_density
            process_summary=summaries[0][1]
            hpd_loss=np.linalg.eigvalsh(-A)[-1]+np.einsum('ni,ij,nj->n',nodes,A,nodes)
            reference_hpd_mass=float(w@(reference_density*(hpd_loss<=process_summary['hpd_threshold'])))
            marginals=high.diagonal_marginals(192,384)
            cdf_error=0.
            for key,target in [('response_lower',.025),('response_upper',.975)]:
                cdf=grid_cdf(192,384,W,w*reference_density,process_summary[key],marginals)
                cdf_error=max(cdf_error,float(np.max(abs(cdf-target))))
            shapes.append(dict(kappa=kappa,trial=trial,t=t,
                process_density_tv=float(.5*w@abs(process_density-reference_density)),
                process_mean_error=float(np.linalg.norm(process_summary['mean']-reference['mean'],2)),
                process_hpd_mass_under_reference=reference_hpd_mass,
                process_scalar_cdf_error=cdf_error,reference_numerical_ok=audit['reference_numerical_ok'],
                inverse_residual=predicted.fit.residual))
    return dict(metrics=rows,query_predictions=queries,reference_checks=audits,
                shape_checks=shapes,predictive_scores=scores)


def wilson(success,n):
    z=1.959963984540054;p=success/n;den=1+z*z/n
    middle=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return middle-half,middle+half


def summarize(metrics,T):
    rows=[]
    for (kappa,method),g in metrics[metrics.t==T].groupby(['kappa','method']):
        n=g.trial.nunique();ball=int(g.ball_covered.sum());hpd=int(g.hpd_covered.sum())
        blo,bhi=wilson(ball,n);hlo,hhi=wilson(hpd,n)
        rows.append(dict(kappa=kappa,method=method,trials=n,numerical_failures=int((~g.numerical_ok).sum()),
            local_approximation_failures=int((~g.local_approximation_ok).sum()),
            ball_coverage=ball/n,ball_low=blo,ball_high=bhi,hpd_coverage=hpd/n,hpd_low=hlo,hpd_high=hhi,
            clean_coverage=float(g.clean_coverage.mean()),clean_width=float(g.clean_width.mean()),
            interval_score=float(g.interval_score.mean()),clean_mse=float(g.clean_mse.mean()),
            posterior_risk=float(g.posterior_risk.mean()),prior_log_score=float(g.prior_log_score.mean())))
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/contact_uncertainty')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--development-only',action='store_true')
    parser.add_argument('--trials',type=int,default=200)
    parser.add_argument('--skip-development',action='store_true',help='Require existing passing development checks')
    parser.add_argument('--development',action='store_true',help='Ungated bounded software smoke only; no scientific claim')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    config=dict(schema=1,development_seed=7315271,test_seed=7315272,kappas=[4.,20.],T=24,s=.12,
        trials=args.trials,reference_trials=min(16,args.trials),report_times=[8,16,24],warmup=6,
        diagnostic_beta=.85,reference_subset='First sixteen independent trajectory identifiers per kappa, fixed before final testing',
        law='Uniform initial unoriented normal, exact Watson kernel, coordinate known inputs; dimensionless, not mechanical contact',
        uncertainty='Clean intervals only; no Gaussian substitute for noisy mixture intervals',
        gaussian='Kernel-matched angular variance; six-step process Bingham warmup, then recursive local Gaussian',
        shape='Same-parameter Laplace changes reporting only; recursive Gaussian also approximates transition and measurement updates',
        reference_resolution=dict(coarse_degree=24,fine_degree=32,coarse_summary_order=128,fine_summary_order=192),
        statistics='Independent trajectories; query/component averages within trajectory; intervals pointwise and exploratory')
    reference_dir=out/'reference';reference_dir.mkdir(exist_ok=True)
    if args.skip_development:
        development=json.loads((reference_dir/'checks.json').read_text())
        for name,digest in development['sources'].items():assert sha(ROOT/name)==digest,name
    else:development=development_check(out,config)
    if args.development_only:return
    if development['status']!='passed':
        raise RuntimeError('Full reference development unresolved; preserve targets and resolve before final tests')
    gate=None
    if not args.development:
        from lib.contact_uncertainty_validation import require_uncertainty_gate
        gate=require_uncertainty_gate(ROOT)
    # Do not retain the large development harmonic caches in the pool parent.
    # Spawned workers build their own fixed reference budgets only as needed.
    grid_basis.cache_clear();query_grid.cache_clear()
    paths=['lib/contact_spherical_reference.py','scripts/run_contact_matched.py','lib/contact_process_prediction.py',
           'lib/contact_gaussian.py','lib/contact_posterior.py','lib/contact_study.py','lib/models.py','lib/geometry.py']
    sources={name:sha(ROOT/name) for name in paths}
    signature=hashlib.sha256(json.dumps(dict(config=config,sources=sources),sort_keys=True).encode()).hexdigest()
    directory=out/'matched';directory.mkdir(exist_ok=True)
    meta=directory/'config.json'
    if meta.exists():
        old=json.loads(meta.read_text())
        if old['source_hash']==signature and all(sha(directory/name)==digest for name,digest in old['tables'].items()):
            print('Matched process: verified cached evidence');return
        raise RuntimeError('Completed matched results differ; preserve and choose a new output directory')
    save_json(directory/'preregistered.json',dict(config=config,sources=sources,source_hash=signature,gate=gate,development_only=args.development,
        development=development,versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)))
    checkpoint=directory/'checkpoints';checkpoint.mkdir(exist_ok=True)
    tasks=[];completed={}
    for kappa in config['kappas']:
        for trial in range(config['trials']):
            key=(kappa,trial);path=checkpoint/f'{int(kappa)}_{trial:04d}_{signature[:12]}.json'
            if path.exists():
                old=json.loads(path.read_text())
                if old['source_hash']==signature and old['result_hash']==hashlib.sha256(json.dumps(old['result'],sort_keys=True).encode()).hexdigest():
                    completed[key]=old['result'];continue
            tasks.append((config,kappa,trial))
    save_json(directory/'status.json',dict(status='running',source_hash=signature,restored=len(completed)))
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn')) as pool:
        futures={pool.submit(trial_job,task):(task[1],task[2]) for task in tasks}
        for future in as_completed(futures):
            key=futures[future];result=portable(future.result());completed[key]=result
            path=checkpoint/f'{int(key[0])}_{key[1]:04d}_{signature[:12]}.json'
            save_json(path,dict(source_hash=signature,result=result,
                result_hash=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()))
            if len(completed)%8==0:print('Matched process',len(completed),'/',2*config['trials'],flush=True)
    frames={name:pd.DataFrame([row for key in sorted(completed) for row in completed[key][name]])
            for name in ['metrics','query_predictions','reference_checks','shape_checks','predictive_scores']}
    frames['summary']=summarize(frames['metrics'],config['T'])
    if any(sha(ROOT/name)!=digest for name,digest in sources.items()):raise RuntimeError('Matched numerical source changed during execution')
    tables={}
    for name,frame in frames.items():
        path=directory/f'{name}.csv';frame.to_csv(path,index=False);tables[path.name]=sha(path)
    checks=dict(trajectories_per_kappa=config['trials'],reference_trajectories_per_kappa=config['reference_trials'],
        reference_targets=len(frames['reference_checks']),
        reference_failures=int((~frames['reference_checks'].reference_numerical_ok).sum()),
        maximum_inverse_residual=float(frames['metrics'].maximum_inverse_residual.max()),
        failures_by_method=frames['metrics'].groupby('method').numerical_ok.apply(lambda g:int((~g).sum())).to_dict(),
        density_projection='Approximate; see shape_checks.csv for full recursive density/region/CDF errors')
    save_json(meta,dict(config=config,sources=sources,source_hash=signature,tables=tables,checks=checks,gate=gate,development_only=args.development,
        workers=args.workers,versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)))
    save_json(directory/'status.json',dict(status='completed',checks=checks,source_hash=signature))
    print(frames['summary'].to_string(index=False));print(json.dumps(checks,indent=2))


if __name__=='__main__':main()

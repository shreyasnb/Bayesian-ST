"""Independent application pilots with frozen task-loss tuning and motion access.

Historical application tables and implementations are left untouched. Validation
uses matched drifting observations; one policy is frozen before robustness tests.
"""
from __future__ import annotations

from itertools import product
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .experiments import _array_data, _field_data, _posterior
from .filter import BinghamFilter
from .geometry import projector, chordal, gaussian, center
from .em import Record
from .models import increment
from .moments import mean_bingham, log_normalizer
from benchmarks.case1 import LoadedCovariance, mvdr, output_power, sample_output_power
from benchmarks.case2 import EWCovariancePCA, RollingPCA, GROUSE

BETAS=(.70,.85,.95,.99,1.)
LOADINGS=(.025,.1,.4,1.6,6.4)
RHOS=(0.,.25,.5,.75,1.)
# Includes values below the old .01 boundary and windows above the old 48.
STEPS=(.0003,.001,.003,.01,.03,.1)
WINDOWS=(12,24,48,96,192)


def candidate_families(application):
    """Cartesian comparator families; different counts are explicitly reported."""
    common={'Posterior mean':[dict(beta=b) for b in BETAS],
            'EW PCA':[dict(beta=b) for b in BETAS],
            'Shrunk PCA':[dict(beta=b,rho=r) for b,r in product(BETAS,RHOS)]}
    if application=='array':
        common['Loaded covariance']=[dict(beta=b,loading=l) for b,l in product(BETAS,LOADINGS)]
    elif application=='field':
        common.update({'Rolling PCA':[dict(window=w) for w in WINDOWS],
                       'GROUSE':[dict(step=s) for s in STEPS]})
    else:raise ValueError('application must be array or field')
    return common


def scenarios():
    return [
        dict(application='array',scenario='Drift',mismatch='Matched',rate=.012,motion='identity',primary=True),
        dict(application='array',scenario='Static',mismatch='Matched',rate=0.,motion='identity',primary=False),
        dict(application='array',scenario='Unequal powers',mismatch='Unequal powers',rate=.012,motion='identity',primary=False),
        dict(application='array',scenario='Correlation',mismatch='Correlation',rate=.012,motion='identity',primary=False),
        dict(application='array',scenario='Contamination',mismatch='Contamination',rate=.012,motion='identity',primary=False),
        dict(application='array',scenario='Partial motion',mismatch='Matched',rate=.012,motion='partial',primary=False),
        dict(application='field',scenario='Matched',mismatch='Matched',rate=.018,motion='identity',primary=True),
        dict(application='field',scenario='Unequal powers',mismatch='Unequal powers',rate=.018,motion='identity',primary=False),
        dict(application='field',scenario='Sparse corruption',mismatch='Sparse corruption',rate=.018,motion='identity',primary=False),
        dict(application='field',scenario='Partial motion',mismatch='Matched',rate=.018,motion='partial',primary=False),
    ]


def _seed(seed,application,trial,role):
    # Distinct validation and test root seeds are still used; no source overlap.
    return int(np.random.SeedSequence([seed,0 if application=='array' else 1,trial,role]).generate_state(1)[0])


def data(config,application,trial,validation=False,scenario=None):
    if scenario is None:
        scenario=next(s for s in scenarios() if s['application']==application and s['primary'])
    seed=config['validation_seed'] if validation else config['test_seed']
    # Scenario IDs intentionally do not enter seeds: robustness cases are paired
    # interventions on a trajectory, not additional independent state draws.
    key=_seed(seed,application,trial,601)
    horizon=config['T_array'] if application=='array' else config['T_field']
    if application=='array':
        U,X,a,R,partial=_array_data(key,horizon,scenario['rate'],.25,scenario['mismatch'])
        n,k,s,tau2=16,3,.3,2.
        rng=np.random.default_rng(_seed(seed,application,trial,602))
        heldout=(np.linalg.cholesky(R[-1])@gaussian(rng,(n,256),True)).T
        result=dict(U=U,X=X,a=a,R=R,heldout=heldout,complex_data=True)
    else:
        Phi,rows,partial=_field_data(key,horizon,drift=scenario['rate'],mismatch=scenario['mismatch'])
        n,k,s,tau2=8,3,.2,1.5
        result=dict(U=np.asarray([r[1] for r in rows]),X=np.asarray([r[0] for r in rows]),
                    clean=np.asarray([r[2] for r in rows]),Phi=Phi,complex_data=False)
    result.update(n=n,k=k,s=s,tau2=tau2,scenario=scenario,
                  G=np.eye(n) if scenario['motion']=='identity' else partial)
    return result


def _is_boundary(parameters,application):
    ranges=dict(beta=BETAS,loading=LOADINGS,rho=RHOS,step=STEPS,window=WINDOWS)
    return any(value in (ranges[key][0],ranges[key][-1]) for key,value in parameters.items())


def _new_tracker(method,p,d,seed):
    if method=='Posterior mean':return BinghamFilter(d['n'],d['k'],d['s'],d['tau2'],beta=p['beta'],complex_data=d['complex_data'])
    if method in ('EW PCA','Shrunk PCA'):return EWCovariancePCA(d['n'],d['k'],beta=p['beta'])
    if method=='Loaded covariance':return LoadedCovariance(d['n'],beta=p['beta'],loading=p['loading'])
    if method=='Rolling PCA':return RollingPCA(d['n'],d['k'],window=p['window'])
    if method=='GROUSE':return GROUSE(d['n'],d['k'],step=p['step'],rng=seed)
    raise ValueError(method)


def _update(tracker,u,G,posterior=False):
    start=perf_counter()
    if posterior:
        # Same fixed-noise recursion as BinghamFilter, instrumented without
        # timing a duplicate prediction or observation correction.
        prior=center(tracker.beta*G@tracker.A@G.conj().T)
        prediction=perf_counter()-start
        start=perf_counter()
        tracker.records.append(Record(prior.copy(),np.asarray(u).copy(),None))
        tracker.A=center(prior+increment(u,tracker.s,tracker.tau2,complex_data=tracker.complex_data))
        tracker.t+=1
        tracker._application_basis=tracker.basis  # Cache the modal update returned by BinghamFilter.
        return perf_counter()-start,prediction
    tracker.transform(G);prediction=perf_counter()-start
    start=perf_counter();tracker.update(u)
    return perf_counter()-start,prediction


def _loss(application,action,d):
    u=d['U'][-1];tau2=d['tau2'];s=d['s']
    start=perf_counter()
    if application=='array':
        w=mvdr(action,d['a'][-1]);heldout=sample_output_power(w,d['heldout'])
        power=output_power(w,d['R'][-1]);loss=heldout
        extras=dict(output_power=power,heldout_power=heldout,sinr_db=-10*np.log10(power))
    else:
        estimate=tau2/(s+tau2)*action@u
        loss=float(np.linalg.norm(estimate-d['clean'][-1])**2)
        extras=dict(field_squared_error=loss,normalized_error=loss/max(float(np.linalg.norm(d['clean'][-1])**2),1e-12))
    return loss,extras,perf_counter()-start


def evaluate_candidate(config,application,trial,method,parameters,validation=False,scenario=None):
    d=data(config,application,trial,validation,scenario)
    seed=config['validation_seed'] if validation else config['test_seed']
    tracker=_new_tracker(method,parameters,d,_seed(seed,application,trial,603))
    update_seconds=prediction_seconds=0.
    for u in d['U']:
        update,prediction=_update(tracker,u,d['G'],method=='Posterior mean')
        update_seconds+=update;prediction_seconds+=prediction
    base=dict(application=application,trial=trial,method=method,parameters=parameters,
              motion=d['scenario']['motion'],scenario=d['scenario']['scenario'],t=len(d['U']),
              primary=d['scenario']['primary'],update_seconds=update_seconds,prediction_seconds=prediction_seconds,
              optimization_seconds=0.,report_frequency='Terminal only',**parameters)
    diagnostics={};moment_seconds=0.;preparation={}
    if method=='Posterior mean':
        basis=tracker._application_basis
        rng=np.random.default_rng(_seed(seed,application,trial,604))
        M,bases,diagnostics=_posterior(tracker.A,d['k'],rng,config['draws'],config['burnin'],d['complex_data'],config['max_draws'],config['max_burnin'])
        moment_seconds=diagnostics['moment_seconds']
        # Numerical equivalence check against matching EW covariance on the same
        # supplied motion. This is not the independently tuned EW competitor.
        ew=EWCovariancePCA(d['n'],d['k'],beta=parameters['beta'])
        for u in d['U']:ew.transform(d['G']);ew.update(u)
        modal_distance=chordal(ew.basis,basis)
        if np.linalg.norm(projector(ew.basis)-projector(basis))>1e-8:
            raise ArithmeticError('Marginal mode no longer matches identically predicted EW PCA')
        start=perf_counter();Pmode=projector(basis);preparation['Mode']=perf_counter()-start
        actions={'Posterior mean':M,'Mode':Pmode}
    elif method=='Loaded covariance':
        start=perf_counter();actions={method:tracker.covariance};preparation[method]=perf_counter()-start;modal_distance=np.nan
    else:
        start=perf_counter();P=projector(tracker.basis)
        actions={method:parameters['rho']*P+(1-parameters['rho'])*d['k']/d['n']*np.eye(d['n']) if method=='Shrunk PCA' else P}
        preparation[method]=perf_counter()-start
        modal_distance=np.nan
    results=[]
    for label,M in actions.items():
        start=perf_counter()
        action=M if method=='Loaded covariance' else d['s']*np.eye(d['n'])+d['tau2']*M if application=='array' else M
        preparation_seconds=perf_counter()-start+preparation.get(label,0.)
        loss,extras,decision_seconds=_loss(application,action,d)
        decision_seconds+=preparation_seconds
        row=dict(base,method=label,loss=loss,
                 decision_seconds=decision_seconds,decision_preparation_seconds=preparation_seconds,ew_modal_distance=modal_distance,**extras,**diagnostics)
        row['moment_seconds']=moment_seconds if label=='Posterior mean' else 0.
        row['cumulative_seconds']=prediction_seconds+update_seconds+row['moment_seconds']+decision_seconds
        results.append(row)
    return results


def validation_job(task):
    config,application,trial,method,parameters=task
    start=perf_counter()
    rows=evaluate_candidate(config,application,trial,method,parameters,True)
    row=rows[0];row['validation_seconds']=perf_counter()-start
    return row


def choose(validation):
    """Freeze one policy per application from matched-drift validation only."""
    table=pd.DataFrame(validation)
    rows=[];selected={}
    for (application,method),g in table.groupby(['application','method']):
        options=[]
        for text,h in g.groupby(g.parameters.map(lambda p:tuple(sorted(p.items())))):
            parameters=dict(text);loss=float(h.loss.mean())
            options.append((loss,parameters))
            rows.append(dict(application=application,method=method,parameters=parameters,validation_loss=loss,
                             validation_trajectories=len(h),candidate_count=g.parameters.map(lambda p:tuple(sorted(p.items()))).nunique(),
                             validation_seconds=float(h.validation_seconds.sum()),selected=False,
                             boundary=_is_boundary(parameters,application),
                             sampling_failures=int((~h.sampling_ok.fillna(True).astype(bool)).sum()) if 'sampling_ok' in h else 0,
                             **parameters))
        loss,p=min(options,key=lambda item:item[0]);selected[application,method]=p
        for row in rows:
            if row['application']==application and row['method']==method and row['parameters']==p:row['selected']=True
    return selected,pd.DataFrame(rows)


def test_job(task):
    config,scenario,trial,selected=task
    rows=[]
    for method in candidate_families(scenario['application']):
        rows.extend(evaluate_candidate(config,scenario['application'],trial,method,selected[scenario['application'],method],False,scenario))
    return rows


def paired_summary(metrics):
    rows=[]
    for (application,scenario),g in metrics.groupby(['application','scenario']):
        table=g.pivot(index='trial',columns='method',values='loss')
        mean=table['Posterior mean']
        for method in table:
            if method=='Posterior mean':continue
            difference=table[method]-mean;n=len(difference)
            sem=float(difference.std(ddof=1)/np.sqrt(n));half=float(student_t.ppf(.975,n-1)*sem)
            flags=g[g.method=='Posterior mean'].sampling_ok.astype(bool)
            rows.append(dict(application=application,scenario=scenario,reference=method,trials=n,
                posterior_loss=float(mean.mean()),baseline_loss=float(table[method].mean()),gain=float(difference.mean()),
                gain_low=float(difference.mean()-half),gain_high=float(difference.mean()+half),
                reduction_percent=float(100*(table[method].mean()-mean.mean())/table[method].mean()),
                posterior_sampling_failures=int((~flags).sum()),interval='Exploratory paired Student 95%; no multiplicity adjustment',
                interpretation='Unsupported positive claim: sampling failures' if (~flags).any() else
                   'Positive paired interval in this pilot' if difference.mean()-half>0 else
                   'Unsuccessful paired interval' if difference.mean()+half<0 else 'Empirically inconclusive'))
    return pd.DataFrame(rows)


def noise_ablation(seed=482193,trials=8,T=32,beta=.95):
    """Separate deterministic rank-one frozen-prior noise-fitting ablation.

    The accepted objective is the frozen-window surrogate, never a claim about
    full state-space likelihood ascent. Learned measurement noise may absorb
    missing motion. This ablation does not alter any main fixed-noise run.
    """
    from scipy.linalg import expm
    rows=[]
    K=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,0.]])
    for rate in (0.,.018,.06):
        for trial in range(trials):
            rng=np.random.default_rng(np.random.SeedSequence([seed,trial,615]))
            x=rng.normal(size=3);x/=np.linalg.norm(x)
            trackers={'Fixed noise':BinghamFilter(3,1,.2,beta=beta,window=8),
                      'Learned noise':BinghamFilter(3,1,.6,beta=beta,window=8)}
            for t in range(1,T+1):
                x=expm(rate*K)@x;v=2*np.eye(3)[:,(t-1)%3];u=projector(x[:,None])@v+np.sqrt(.2)*rng.normal(size=3)
                for label,f in trackers.items():
                    fitting=label=='Learned noise' and t%8==0
                    before=f.A.copy();start=perf_counter()
                    f.update(u,v=v,fit=fitting,mean=mean_bingham if fitting else None,iterations=4,
                             log_normalizer=lambda A:log_normalizer(A,1))
                    expected=f.beta*before
                    if np.linalg.norm(f.records[-1].prior-expected)>1e-8:raise ArithmeticError('Stored prior indexing changed')
                    if np.linalg.norm(f.A-(expected+increment(u,f.s,None,v)-np.trace(expected+increment(u,f.s,None,v))/3*np.eye(3)))>1e-8:
                        raise ArithmeticError('Noise fit duplicated a likelihood correction')
                    if t%8==0:
                        fit=f.last_fit if fitting else None
                        rows.append(dict(rate=rate,trial=trial,t=t,method=label,s=f.s,true_s=.2,beta=beta,
                            chordal_error=float(chordal(f.basis,x[:,None])),fit_seconds=perf_counter()-start,
                            stored_window=len(f.records),accepted=fit.accepted if fit else 0,
                            rejected=fit.rejected if fit else 0,
                            frozen_objective_change=fit.objectives[-1]-fit.objectives[0] if fit else np.nan,
                            frozen_ascent_checked=bool(fitting),state_space_ascent_claim=False))
    return pd.DataFrame(rows)

"""Bounded process-model pilot with an independent full-density circle reference.

The main inference backend is deterministic. Fourier convolution propagates the
entire real Gr(1,2) density as a separate reference, rather than declaring a
small inverse-moment residual a certificate for credible-region accuracy.
"""
from pathlib import Path
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import sys,json,hashlib,platform,argparse
from collections import deque
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import scipy
from scipy.special import logsumexp,ndtr
from scipy.optimize import brentq
from lib.process_prediction import (moment_and_normalizer,transition_coefficients,
                                    invert_bingham_mean,stochastic_predict)
from lib.spectral_series import rank_one_bingham
from lib.models import increment
from lib.geometry import center,mode,projector,chordal,risk
from lib.em import Record,fit_noise


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grid(N):
    theta=np.pi*np.arange(N)/N
    X=np.column_stack([np.cos(theta),np.sin(theta)])
    return theta,np.einsum('bi,bj->bij',X,X)


def density(A,P):
    energy=np.einsum('ij,bji->b',A,P)
    return np.exp(energy-logsumexp(energy))


def kernel(kappa,theta):
    e=(kappa/2)*np.cos(2*theta)
    return np.exp(e-logsumexp(e))


def predict_mass(mass,transition_fft):
    # Roundoff may produce tiny negative probabilities, never physical mass.
    predicted=np.fft.irfft(np.fft.rfft(mass)*transition_fft,n=len(mass))
    if predicted.min() < -1e-12:raise ArithmeticError('Convolution lost positivity')
    predicted=np.maximum(predicted,np.finfo(float).tiny)
    return predicted/predicted.sum()


def condition(mass,B,P):
    e=np.einsum('ij,bji->b',B,P)
    w=mass*np.exp(e-e.max())
    return w/w.sum()


def weighted_quantile(values,mass,q):
    order=np.argsort(values);cdf=np.cumsum(mass[order]);cdf[-1]=1.
    return float(values[order[min(np.searchsorted(cdf,q),len(order)-1)]])


def interval_summary(mass,theta,P,true_theta,s,future_observation,A=None):
    N=len(theta)
    if A is None:
        centre=float(theta[np.argmax(mass)])
        loss=np.log(mass.max())-np.log(np.maximum(mass,np.finfo(float).tiny))
        pos=(true_theta%np.pi)*N/np.pi;j=int(np.floor(pos))%N;fraction=pos-np.floor(pos)
        logtrue=(1-fraction)*np.log(max(mass[j],np.finfo(float).tiny))+fraction*np.log(max(mass[(j+1)%N],np.finfo(float).tiny))
        true_loss=float(np.log(mass.max())-logtrue)
    else:
        X=mode(A,1);centre=float(np.arctan2(X[1,0],X[0,0])%np.pi)
        top=float(np.linalg.eigvalsh(A)[-1]);loss=top-np.einsum('ij,bji->b',A,P)
        x=np.array([np.cos(true_theta),np.sin(true_theta)])
        true_loss=float(top-x@A@x)
    distances=abs(np.sin(theta-centre));radius=weighted_quantile(distances,mass,.95)
    threshold=weighted_quantile(loss,mass,.95)
    response=P[:,1,0];clean=float(.5*np.sin(2*true_theta))
    low=weighted_quantile(response,mass,.025);high=weighted_quantile(response,mass,.975)
    def cdf(x):return float(mass@ndtr((x-response)/np.sqrt(s)))
    bracket=(-.5-9*np.sqrt(s),.5+9*np.sqrt(s))
    noisy_low=brentq(lambda x:cdf(x)-.025,*bracket,xtol=1e-10)
    noisy_high=brentq(lambda x:cdf(x)-.975,*bracket,xtol=1e-10)
    return dict(ball_covered=bool(abs(np.sin(true_theta-centre))<=radius+1e-12),
                hpd_covered=bool(true_loss<=threshold+1e-12),
                clean_covered=bool(low<=clean<=high),noisy_covered=bool(noisy_low<=future_observation<=noisy_high),
                ball_radius=radius,ball_width=float(2*np.arcsin(min(1.,radius))),
                hpd_threshold=threshold,hpd_width=float(np.mean(loss<=threshold)*np.pi),
                hpd_probability_under_maintained=float(mass[loss<=threshold].sum()),
                clean_lower=low,clean_upper=high,clean_width=high-low,
                noisy_lower=noisy_low,noisy_upper=noisy_high,noisy_width=noisy_high-noisy_low,
                threshold_mcse=0.,numerical_method='deterministic periodic grid; resolution assessed separately')


def reference_checks():
    results={};A=np.array([[4.,.8],[.8,-4.]])
    G=np.array([[np.cos(.1),-np.sin(.1)],[np.sin(.1),np.cos(.1)]])
    refs=[]
    for N in (1024,4096):
        theta,P=grid(N);old=density(A,P);K=np.fft.rfft(kernel(10.,theta))
        exact=predict_mass(old,K)
        predicted=stochastic_predict(A,1,10.)
        projected=density(predicted.prior,P)
        moment=np.einsum('b,bij->ij',exact,P)
        error=float(np.linalg.norm(moment-predicted.target_mean,2))
        assert error<1e-11
        B=increment(np.array([.7,-.2]),.12,v=np.array([1.,0.]))
        exact_post=condition(exact,B,P);Apost=center(predicted.prior+B)
        projected_post=density(Apost,P)
        summary=interval_summary(projected_post,theta,P,.2,.12,.3,Apost)
        loss=float(np.linalg.eigvalsh(Apost)[-1])-np.einsum('ij,bji->b',Apost,P)
        region_mass=float(exact_post[loss<=summary['hpd_threshold']].sum())
        refs.append(dict(grid=N,first_moment_error=error,fit_residual=predicted.fit.residual,
                         prediction_tv=float(abs(exact-projected).sum()/2),
                         posterior_tv=float(abs(exact_post-projected_post).sum()/2),
                         projected_hpd_mass_under_exact_posterior=region_mass,
                         hpd_threshold=summary['hpd_threshold'],ball_radius=summary['ball_radius']))
    # Absolute differences introduce integration kinks, unlike smooth moments.
    assert abs(refs[0]['prediction_tv']-refs[1]['prediction_tv'])<1e-6
    assert abs(refs[0]['posterior_tv']-refs[1]['posterior_tv'])<1e-6
    assert abs(refs[0]['hpd_threshold']-refs[1]['hpd_threshold'])<.015
    assert refs[1]['prediction_tv']>.001,'Reference must expose non-Bingham closure'
    results['full_density_reference']=refs
    results['interpretation']='Exact predicted moment conditional on the maintained previous Bingham; full recursive filter is approximate.'
    rng=np.random.default_rng(871305);N=100000;kappa=6.
    samples=[]
    theta=rng.vonmises(0,kappa/2,N)/2
    real=np.column_stack([np.cos(theta),np.sin(theta)])
    t=np.log1p(rng.random(N)*np.expm1(kappa))/kappa
    complex_samples=np.column_stack([np.sqrt(t),np.sqrt(1-t)*np.exp(2j*np.pi*rng.random(N))])
    for field,X in [('real',real),('complex',complex_samples)]:
        c=transition_coefficients(2,1,kappa,field=='complex')
        projectors=np.einsum('bi,bj->bij',X,X.conj())
        mean=projectors.mean(axis=0);se=np.std(projectors.real,axis=0,ddof=1)/np.sqrt(N)
        error=float(np.max(abs(mean-np.diag([c.a,c.b]))))
        assert error<.006
        samples.append(dict(field=field,kappa=kappa,draws=N,mean_error=error,maximum_real_entry_iid_se=float(se.max())))
    results['independent_transition_draws']=samples
    comparison=[]
    for d in (0.,1.,20.,200.):
        B=np.diag([d,-d]);M,psi,_=moment_and_normalizer(B)
        r=rank_one_bingham(B,tol=1e-12)
        error=max(float(abs(psi-r.log_normalizer)),float(np.max(abs(M-r.mean))))
        assert error<2e-10
        comparison.append(dict(span=2*d,analytic_series_error=error))
    results['circle_analytic_vs_series']=comparison
    high=transition_coefficients(3,1,1000.)
    assert high.rho>.995
    results['large_kappa']=dict(kappa=1000.,n=3,k=1,rho=high.rho)
    try:transition_coefficients(8,3,10.)
    except NotImplementedError:results['general_rank_status']='unresolved; explicitly rejected'
    else:raise AssertionError('An unsupported general rank must not silently proceed')
    return results


def one_trial(law,kappa,trial,seed,T=24,s=.12,N=2048):
    theta,P=grid(N);K=np.fft.rfft(kernel(kappa,theta));rng=np.random.default_rng(np.random.SeedSequence([seed,trial,6091]))
    truth=float(rng.uniform(0,np.pi));As={name:np.zeros((2,2)) for name in ['Projected transition','Identity','Power forgetting']}
    full=np.ones(N)/N;reports=[];innovations=[];paths=[]
    clocks={name:dict(prediction=0.,correction=0.,moment=0.,inversion=0.,region=0.) for name in [*As,'Full-density reference']}
    for t in range(1,T+1):
        if law=='Matched transition':truth+=rng.vonmises(0,kappa/2)/2
        elif law=='Slow deterministic':truth+=.015
        elif law=='Fast deterministic':truth+=.10
        elif law=='Acceleration and turn':truth+=(.008*t if t<=T//2 else -.12)
        elif law=='Observation contamination':truth+=rng.vonmises(0,kappa/2)/2
        else:raise ValueError(law)
        truth%=np.pi;x=np.array([np.cos(truth),np.sin(truth)]);Pt=np.outer(x,x)
        v=np.eye(2)[:,(t-1)%2]
        u=Pt@v+np.sqrt(s)*rng.normal(size=2)
        if law=='Observation contamination' and rng.random()<.12:u+=1.5*rng.normal(size=2)
        B=increment(u,s,v=v)
        begin=perf_counter();reference_prior=predict_mass(full,K)
        clocks['Full-density reference']['prediction']+=perf_counter()-begin
        begin=perf_counter();reference_prior_mean=np.einsum('b,bij->ij',reference_prior,P)
        clocks['Full-density reference']['moment']+=perf_counter()-begin
        begin=perf_counter();full=condition(reference_prior,B,P)
        clocks['Full-density reference']['correction']+=perf_counter()-begin
        priors={};means={};fit_residual=0.;fit_uncertainty=0.
        for name,A in list(As.items()):
            begin=perf_counter()
            if name=='Projected transition':
                prediction=stochastic_predict(A,1,kappa)
                prior=prediction.prior;Mp=prediction.target_mean
                clocks[name]['inversion']+=prediction.fit.seconds
                fit_residual=prediction.fit.residual;fit_uncertainty=prediction.fit.moment_uncertainty
                if t in (8,16,T):same_prior_exact=predict_mass(density(A,P),K)
            else:
                prior=center((.85 if name=='Power forgetting' else 1.)*A)
                Mp=moment_and_normalizer(prior)[0]
            clocks[name]['prediction']+=perf_counter()-begin
            priors[name]=prior;means[name]=Mp
            begin=perf_counter();As[name]=center(prior+B)
            clocks[name]['correction']+=perf_counter()-begin
        means['Full-density reference']=reference_prior_mean
        for name,Mp in means.items():
            residual=u-Mp@v;expected=2*s+risk(Mp,v)
            innovations.append(dict(law=law,kappa=kappa,trial=trial,t=t,method=name,
                residual0=float(residual[0]),residual1=float(residual[1]),
                residual_energy=float(residual@residual),expected_energy=float(expected)))
        if trial==0:
            for name,A in As.items():
                M=moment_and_normalizer(A)[0];X=mode(A,1)
                paths.append(dict(law=law,kappa=kappa,t=t,method=name,true_response=float(Pt[1,0]),
                                  mean_response=float(M[1,0]),mode_error=chordal(X,x[:,None])))
        if t not in (8,16,T):continue
        future=float(Pt[1,0]+np.sqrt(s)*rng.normal())
        same_prior_post=condition(same_prior_exact,B,P)
        for name in [*As,'Full-density reference']:
            begin=perf_counter()
            A=None if name=='Full-density reference' else As[name]
            mass=full if A is None else density(A,P)
            info=interval_summary(mass,theta,P,truth,s,future,A)
            clocks[name]['region']+=perf_counter()-begin
            if A is None:
                tv=0.;one_step_tv=0.;prior_tv=0.;hpd_reference_mass=info['hpd_probability_under_maintained']
                mean_error=0.;modal_error=abs(np.sin(theta[np.argmax(full)]-truth))
            else:
                tv=float(abs(mass-full).sum()/2)
                mean_error=float(np.linalg.norm(moment_and_normalizer(A)[0]-np.einsum('b,bij->ij',full,P),2))
                modal_error=chordal(mode(A,1),x[:,None])
                loss=float(np.linalg.eigvalsh(A)[-1])-np.einsum('ij,bji->b',A,P)
                hpd_reference_mass=float(full[loss<=info['hpd_threshold']].sum())
                one_step_tv=float(abs(mass-same_prior_post).sum()/2) if name=='Projected transition' else np.nan
                prior_tv=float(abs(density(priors[name],P)-same_prior_exact).sum()/2) if name=='Projected transition' else np.nan
            reports.append(dict(law=law,kappa=kappa,trial=trial,t=t,method=name,**info,
                modal_error=float(modal_error),reference_density_tv=tv,reference_mean_error=mean_error,
                one_step_projection_tv=one_step_tv,one_step_prior_projection_tv=prior_tv,
                hpd_mass_under_reference=hpd_reference_mass,
                inverse_moment_residual=fit_residual if name=='Projected transition' else 0.,
                inverse_moment_uncertainty=fit_uncertainty if name=='Projected transition' else 0.,
                prediction_seconds=clocks[name]['prediction'],correction_seconds=clocks[name]['correction'],
                moment_seconds=clocks[name]['moment'] if name=='Full-density reference' else np.nan,
                inversion_seconds=clocks[name]['inversion'],region_seconds=clocks[name]['region'],
                moment_timing_scope='separate' if name=='Full-density reference' else 'embedded in prediction/inversion',
                scalar_response_lower=info['clean_lower'],scalar_response_upper=info['clean_upper']))
    return reports,innovations,paths


def noise_ablation(trials=20,T=24,s=.12,kappa=4.,seed=871406):
    rows=[]
    def mean(A,k):return moment_and_normalizer(A,k)[0]
    def logz(A):return moment_and_normalizer(A)[1]
    for trial in range(trials):
        rng=np.random.default_rng(np.random.SeedSequence([seed,trial,6091]))
        theta=float(rng.uniform(0,np.pi));A=np.zeros((2,2));records=deque(maxlen=8);estimate=.24
        for t in range(1,T+1):
            theta+=rng.vonmises(0,kappa/2)/2;x=np.array([np.cos(theta),np.sin(theta)])
            v=np.eye(2)[:,(t-1)%2];u=x*(x@v)+np.sqrt(s)*rng.normal(size=2)
            prior=stochastic_predict(A,1,kappa).prior
            records.append(Record(prior.copy(),u.copy(),v.copy()))
            if t%8==0:
                fit=fit_noise(list(records),1,estimate,None,mean,iterations=5,log_normalizer=logz)
                assert np.min(np.diff(fit.objectives))>=-1e-8
                estimate=fit.s
                rows.append(dict(trial=trial,t=t,true_s=s,estimated_s=estimate,
                                 first_objective=fit.objectives[0],last_objective=fit.objectives[-1],
                                 accepted=fit.accepted,rejected=fit.rejected,window=len(records),
                                 full_state_space_ascent='unverified; safeguard is for frozen-prior surrogate only'))
            A=center(prior+increment(u,estimate,v=v))
    return pd.DataFrame(rows)


def wilson(success,n):
    z=1.959963984540054;p=success/n;den=1+z*z/n
    centre=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return float(centre-half),float(centre+half)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results/dev_prediction/process')
    parser.add_argument('--quick',action='store_true',help='Software smoke only: 4 matched and 2 stress trajectories')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    config=dict(seed=871306,T=24,s=.12,grid=2048,matched_trials=4 if args.quick else 200,
                stress_trials=2 if args.quick else 40,kappas=[4.,20.],report_times=[8,16,24],
                forgetting=.85,fixed_noise=True,primary='Matched transition at terminal time; each kappa is preselected',
                method='Real Gr(1,2), deterministic Bessel inverse; independent Fourier full-density reference',
                theoretical_scope='Exact first moment conditional on maintained previous Bingham; assumed-density full recursion',
                inference_claim='No universal calibration claim; matched evidence and reference differences are measured',
                multiplicity='Coverage intervals are pointwise Wilson intervals; additional stress results are exploratory')
    # Freeze the complete protocol and seeds before inspecting any final trial.
    config['regimes']=[dict(law='Matched transition',kappa=k,trials=config['matched_trials'],seed=config['seed']+int(k)*10) for k in config['kappas']]
    config['regimes'].extend(dict(law=law,kappa=4.,trials=config['stress_trials'],seed=config['seed']+1000+i*100) for i,law in enumerate(
        ['Slow deterministic','Fast deterministic','Acceleration and turn','Observation contamination']))
    config_path=out/'config.json'
    config_path.write_text(json.dumps(config,indent=2))
    checks=reference_checks();(out/'reference_checks.json').write_text(json.dumps(checks,indent=2))
    rows=[];innov=[];paths=[];start=perf_counter()
    regimes=[(r['law'],r['kappa'],r['trials'],r['seed']) for r in config['regimes']]
    for law,kappa,trials,seed in regimes:
        for trial in range(trials):
            a,b,c=one_trial(law,kappa,trial,seed,config['T'],config['s'],config['grid'])
            rows.extend(a);innov.extend(b);paths.extend(c)
        print(f'Process pilot: {law}, kappa={kappa}, trajectories={trials}',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(out/'coverage_trials.csv',index=False)
    innovation=pd.DataFrame(innov);innovation.to_csv(out/'innovations.csv',index=False)
    pd.DataFrame(paths).to_csv(out/'paths.csv',index=False)
    noise_ablation(2 if args.quick else 20).to_csv(out/'noise_ablation.csv',index=False)
    terminal=frame[frame.t==config['T']];summary=[]
    for (law,kappa,method),g in terminal.groupby(['law','kappa','method']):
        row=dict(law=law,kappa=kappa,method=method,trials=len(g),mean_modal_error=float(g.modal_error.mean()),
                 mean_reference_tv=float(g.reference_density_tv.mean()),
                 mean_hpd_mass_under_reference=float(g.hpd_mass_under_reference.mean()),
                 maximum_inverse_residual=float(g.inverse_moment_residual.max()),
                 mean_prediction_seconds=float(g.prediction_seconds.mean()),
                 mean_correction_seconds=float(g.correction_seconds.mean()),
                 mean_moment_seconds=float(g.moment_seconds.mean()),mean_region_seconds=float(g.region_seconds.mean()),
                 mean_inversion_seconds=float(g.inversion_seconds.mean()))
        for kind,width in [('ball','ball_width'),('hpd','hpd_width'),('clean','clean_width'),('noisy','noisy_width')]:
            success=int(g[kind+'_covered'].sum());low,high=wilson(success,len(g))
            row.update({kind+'_covered':success,kind+'_coverage':success/len(g),kind+'_wilson_low':low,
                        kind+'_wilson_high':high,kind+'_mean_width':float(g[width].mean())})
        h=innovation[(innovation.law==law)&(innovation.kappa==kappa)&(innovation.method==method)]
        correlations=[]
        for _,tr in h.groupby('trial'):
            for key in ['residual0','residual1']:
                correlations.append(np.corrcoef(tr[key].to_numpy()[:-1],tr[key].to_numpy()[1:])[0,1])
        row.update(innovation_energy=float(h.residual_energy.mean()),innovation_expected=float(h.expected_energy.mean()),
                   innovation_mean_lag1=float(np.nanmean(correlations)))
        summary.append(row)
    pd.DataFrame(summary).to_csv(out/'coverage_summary.csv',index=False)
    sources={p:sha(ROOT/p) for p in ['lib/process_prediction.py','scripts/check_process_prediction.py','lib/models.py','lib/spectral_series.py','lib/em.py']}
    tables={p.name:sha(p) for p in out.glob('*.csv')}
    status=dict(status='passed numerical reference checks; pilot outcomes retained',general_rank='unresolved',
                config_sha256=sha(config_path),sources=sources,tables=tables,
                versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
                seconds=perf_counter()-start,hardware=platform.machine(),
                statistical_replicates='Independent trajectory identifiers, not posterior/grid draws',
                projection='Moment residual does not bound density TV, HPD mass, or accumulated filtering error',
                timing='Cumulative at reporting time; prediction includes moments and inversion. Inversion is nested, not additive. Standalone moment timing only for full-density reference; embedded timings are missing rather than zero.',
                noise_fit='Separate 20-trajectory frozen-prior surrogate ablation; full state-space ascent unverified')
    (out/'status.json').write_text(json.dumps(status,indent=2))
    print(pd.DataFrame(summary)[['law','kappa','method','ball_coverage','hpd_coverage','clean_coverage','noisy_coverage','mean_reference_tv']].to_string(index=False))


if __name__=='__main__':main()

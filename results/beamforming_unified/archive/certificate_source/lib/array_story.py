"""Native-complex n=16 prediction spectra and beamforming uncertainty.

Uses fixed preassigned application trajectories, not a selected winning example.
Posterior bands are pointwise in steering angle, not simultaneous bands.
"""
import numpy as np
import pandas as pd
from .experiments import _array_data,_posterior
from .filter import BinghamFilter
from .geometry import projector
from benchmarks.case1 import LoadedCovariance,mvdr,output_power


def run_array_story(metrics,tuning,seed=1729,T=72,draws=6000,burnin=2000,
                    max_draws=24000,max_burnin=8000):
    spectra=[];power=[];diagnostics=[]
    angles=np.linspace(-70.,70.,281)
    steering=np.exp(1j*np.pi*np.outer(np.arange(16),np.sin(np.deg2rad(angles))))/4
    for scenario,case,drift in [('Static',0,0.),('Drift',3,.012)]:
        m=metrics[(metrics.scenario==scenario)&(metrics.method=='Posterior mean')]
        beta=float(m.beta.iloc[0])
        tune=tuning[(tuning.scenario==scenario)&(tuning.method=='Loaded covariance')]
        tune=tune.loc[tune.validation_loss.idxmin()]
        U,X,a,R,G=_array_data(seed+100*case,T,drift,.25,'Matched')
        f=BinghamFilter(16,3,.3,2.,beta=beta,complex_data=True)
        loaded=LoadedCovariance(16,beta=float(tune.beta),loading=float(tune.loading))
        rng=np.random.default_rng(np.random.SeedSequence([seed,case,6143]))
        for t,u in enumerate(U,1):
            f.update(u,G=G);loaded.transform(G);cov=loaded.update(u)
            if t not in (12,36,T):continue
            M,bases,diag=_posterior(f.A,3,rng,draws,burnin,True,max_draws,max_burnin)
            # Work in chunks so a full N x k x angle complex array is not retained.
            angular=np.empty((len(bases),len(angles)))
            for start in range(0,len(bases),256):
                block=bases[start:start+256]
                coefficients=np.einsum('bni,nj->bij',block.conj(),steering,optimize=True)
                angular[start:start+len(block)]=.3+2*np.sum(abs(coefficients)**2,axis=1)
            qs=np.quantile(angular,[.025,.25,.5,.75,.975],axis=0)
            truth=.3+2*np.sum(abs(X[t-1].conj().T@steering)**2,axis=0)
            mode=.3+2*np.sum(abs(f.basis.conj().T@steering)**2,axis=0)
            predicted=.3+2*np.einsum('ij,ij->j',steering.conj(),M@steering).real
            for j,angle in enumerate(angles):
                spectra.append(dict(scenario=scenario,t=t,angle=angle,truth=truth[j],mean=predicted[j],mode=mode[j],
                    q025=qs[0,j],q25=qs[1,j],q50=qs[2,j],q75=qs[3,j],q975=qs[4,j],sampling_ok=diag['sampling_ok']))
            for method,C in [('Posterior mean',.3*np.eye(16)+2*M),('Mode',.3*np.eye(16)+2*projector(f.basis)),('Loaded covariance',cov)]:
                w=mvdr(C,a[t-1]);cost=.3*np.vdot(w,w).real+2*np.sum(abs(np.einsum('bni,n->bi',bases.conj(),w))**2,axis=1)
                actual=output_power(w,R[t-1])
                for j in np.linspace(0,len(cost)-1,min(1024,len(cost)),dtype=int):
                    power.append(dict(scenario=scenario,t=t,method=method,draw=int(j),power=float(cost[j]),true_power=actual,posterior_expected_power=float(np.mean(cost))))
            diagnostics.append(dict(scenario=scenario,t=t,**diag))
            print(f'Array uncertainty: {scenario}, t={t}, ESS={diag["min_ess"]:.1f}, passed={diag["sampling_ok"]}',flush=True)
    return dict(spectra=pd.DataFrame(spectra),power=pd.DataFrame(power),diagnostics=pd.DataFrame(diagnostics),
                config=dict(seed=seed,T=T,n=16,k=3,complex_data=True,draws=draws,burnin=burnin,max_draws=max_draws,max_burnin=max_burnin,
                    trajectories='Preassigned repeat zero, Static and Drift; same data/tuning as original application',
                    quantity='Interference-plus-noise power along physical ULA steering directions',
                    intervals='Pointwise posterior quantiles in angle, not simultaneous or frequentist guarantees',
                    caveat='The original desired reference vector is prescribed, not generally a single-angle steering vector; no fictitious desired angle is drawn'))

"""Execute the guide's algebra/Fisher references without fitting a tracker."""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import scipy
from lib.beamforming_theory import passive_fisher_information, passive_scores
from lib.guide_information import (algebra_checks, chart_basis, conditional_fisher,
    conditional_scores, expected_nll, finite_hessian, gaussian_kl,
    independent_moments, native_moments, score_statistics)
from lib.models import increment
from lib.theory import fisher_information

SOURCES = ['lib/guide_information.py', 'scripts/run_guide_information.py',
    'tests/test_guide_information.py', 'lib/models.py', 'lib/filter.py',
    'lib/beamforming_theory.py', 'lib/spectral_series.py', 'lib/theory.py']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configuration(profile):
    return dict(profile=profile, source_commit='6e78cc9911e7e78452633f13dd947afa669c3470',
        seed=2026110901, algebra_seed=2026110902,
        cases=[dict(n=3, k=1, complex_data=False), dict(n=3, k=2, complex_data=False),
               dict(n=5, k=2, complex_data=False), dict(n=3, k=1, complex_data=True),
               dict(n=5, k=2, complex_data=True)],
        sensor_variances=[.2, 1., 4.], signal_variance=1.3,
        score_episodes=8192 if profile == 'smoke' else 65536,
        discounted_episodes=4096 if profile == 'smoke' else 32768,
        finite_difference_steps=[.03, .01, .003, .001],
        curvature_datasets=12 if profile == 'smoke' else 128,
        curvature_observations=[1, 4, 16],
        quadrature_orders=[32, 64, 96], native_tolerance=1e-11,
        betas=[1., .8], weighted_observations=6,
        thresholds=dict(maximum_product_z=7., maximum_score_mean_z=7.,
            final_fd_relative_error=5e-5, likelihood_absolute=2e-9,
            reference_resolution=2e-9, native_reference=2e-9),
        variance_convention='s and tau2 are variances; each real/imag noise coordinate has variance s/2 for proper complex data',
        coordinate_order='vec_F Z (real), [Re vec_F Z, Im vec_F Z] (complex); identical canonical tangent basis for all matrix comparisons',
        observation_scope='Known truth evaluates reference information only. Conditional inputs are independently prescribed, never reconstructed from measured vectors.',
        score_scope='Each repeated score is one independent dataset; products have reported Monte Carlo SEs, not deterministic confidence enclosures.',
        curvature_scope='Realized posterior curvature at its own mode is 2 times selected/complementary eigenvalue gaps. Expected information is at fixed truth; only spectra are compared across these different tangent frames.')


def normal(rng, shape, complex_data):
    real = rng.normal(size=shape)
    return (real+1j*rng.normal(size=shape))/np.sqrt(2) if complex_data else real


def record_statistics(scores, F, base, matrices, checks):
    empirical, mean, mcse = score_statistics(scores)
    tiny = max(float(np.max(np.diag(F))), 1.)*1e-13
    z = abs(empirical-F)/np.maximum(mcse, tiny)
    mean_z = abs(mean)/np.maximum(np.sqrt(np.diag(F)/len(scores)), tiny)
    for i in range(len(F)):
        for j in range(len(F)):
            matrices.append(dict(**base, i=i, j=j, expected=float(F[i, j]),
                score_covariance=float(empirical[i, j]), product_mcse=float(mcse[i, j])))
    passed = bool(z.max() < 7 and mean_z.max() < 7)
    checks.append(dict(**base, dimension=len(F), draws=len(scores),
        relative_matrix_error=float(np.linalg.norm(empirical-F)/np.linalg.norm(F)),
        maximum_product_z=float(z.max()), maximum_score_mean_z=float(mean_z.max()),
        expected_min=float(np.linalg.eigvalsh(F)[0]), expected_max=float(np.linalg.eigvalsh(F)[-1]),
        status='passed' if passed else 'inconclusive'))


def run(cfg, out):
    rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], 0 if cfg['profile']=='smoke' else 1]))
    matrices, checks, finite, algebra, curvature, weighted, reference = [], [], [], [], [], [], []
    count, q = cfg['score_episodes'], cfg['signal_variance']
    for idx, item in enumerate(cfg['cases']):
        n, k, cx = item['n'], item['k'], item['complex_data']
        X, Q = np.eye(n)[:, :k], np.eye(n)[:, k:]
        dim = (2 if cx else 1)*k*(n-k)
        for s in cfg['sensor_variances']:
            case = f"{'complex' if cx else 'real'}_N{n}_k{k}_s{s:g}"
            # This full-rank diagonal design is fixed before any measurements.
            V = np.diag(np.linspace(.3, 1.3, n))
            for channel in ('marginal', 'conditional'):
                known = None if channel == 'marginal' else V
                base = dict(case=case, n=n, k=k, complex_data=cx, channel=channel, s=s, tau2=q)
                if known is None:
                    U = np.sqrt(q)*normal(rng, (count, k), cx)@X.T+np.sqrt(s)*normal(rng, (count, n), cx)
                    scores = passive_scores(U, X, s, q, cx, complement=Q)
                    F = passive_fisher_information(n, k, s, q, complex_data=cx)
                else:
                    noise = np.sqrt(s)*normal(rng, (count, len(V), n), cx)
                    scores = conditional_scores(noise, X, V, s, cx, Q)
                    F = conditional_fisher(X, V, s, cx, Q)
                    if not cx:
                        error = np.max(abs(F-fisher_information(X, V.T, s)))
                        algebra.append(dict(case=case, check='conditional_recorded_V_formula', error=error,
                            tolerance=2e-10, status='passed' if error < 2e-10 else 'failed'))
                record_statistics(scores, F, base, matrices, checks)
                for step in cfg['finite_difference_steps']:
                    H = finite_hessian(lambda z: expected_nll(X, z, s, q, cx, known, Q), dim, step)
                    K = finite_hessian(lambda z: gaussian_kl(X, z, s, q, cx, known, Q), dim, step)
                    for i in range(dim):
                        for j in range(dim):
                            finite.append(dict(**base, step=step, i=i, j=j, fisher=F[i,j],
                                negative_expected_hessian=H[i,j], kl_hessian=K[i,j]))
            # Realized finite-data curvature is retained without treating its
            # non-isotropy as non-isotropy of the expected marginal information.
            for dataset in range(cfg['curvature_datasets']):
                U = np.sqrt(q)*normal(rng, (max(cfg['curvature_observations']), k), cx)@X.T
                U += np.sqrt(s)*normal(rng, U.shape, cx)
                A = np.zeros((n,n), dtype=complex if cx else float)
                for t, u in enumerate(U, 1):
                    A += increment(u, s, q, complex_data=cx)
                    if t in cfg['curvature_observations']:
                        eigen = np.linalg.eigvalsh(A)[::-1]
                        gaps = 2*(eigen[:k][None, :]-eigen[k:][:, None]).ravel(order='F')
                        if cx:
                            gaps = np.r_[gaps, gaps]
                        for j, value in enumerate(gaps):
                            curvature.append(dict(case=case, dataset=dataset, n=n, k=k, complex_data=cx,
                                s=s, tau2=q, t=t, index=j, realized_curvature=value,
                                expected_information=t*(2 if cx else 1)*q*q/(s*(s+q))))
        print('Information', idx+1, '/', len(cfg['cases']), flush=True)
    for cx, k in [(False,1), (False,2), (True,1)]:
        for row in algebra_checks(np.random.default_rng(cfg['algebra_seed']+int(cx)+10*k), cx, k):
            algebra.append(dict(case=f"{'complex' if cx else 'real'}_N3_k{k}", **row))
        raw = normal(rng, (3,3), cx)
        A = .7*(raw+raw.conj().T)
        exact_M, exact_z = native_moments(A, k, cx, cfg['native_tolerance'])
        previous = None
        for order in cfg['quadrature_orders']:
            M, z = independent_moments(A, k, cx, order)
            reference.append(dict(complex_data=cx, n=3, k=k, order=order,
                mean_error=float(np.linalg.norm(M-exact_M,2)), logz_error=abs(z-exact_z),
                mean_resolution=np.nan if previous is None else float(np.linalg.norm(M-previous[0],2)),
                logz_resolution=np.nan if previous is None else abs(z-previous[1]),
                measure='complex uniform simplex' if cx else 'real normalized sphere'))
            previous = M, z
    # Independent prescribed mixture: channels are chosen before observations.
    for cx in (False, True):
        n,k,s,T = 3,1,.7,cfg['weighted_observations']
        X, Q = np.eye(n)[:, :k], np.eye(n)[:, k:]
        V = np.eye(n)[np.arange(T)%n]*np.linspace(.4,1.4,T)[:,None]
        dimension = (2 if cx else 1)*(n-1)
        channels = ['marginal' if j%2==0 else 'conditional' for j in range(T)]
        all_scores, all_F = [], []
        episodes = cfg['discounted_episodes']
        for j, channel in enumerate(channels):
            if channel == 'marginal':
                U = np.sqrt(q)*normal(rng,(episodes,k),cx)@X.T+np.sqrt(s)*normal(rng,(episodes,n),cx)
                all_scores.append(passive_scores(U,X,s,q,cx,complement=Q))
                all_F.append(passive_fisher_information(n,k,s,q,complex_data=cx))
            else:
                noise = np.sqrt(s)*normal(rng,(episodes,1,n),cx)
                all_scores.append(conditional_scores(noise,X,V[j:j+1],s,cx,Q))
                all_F.append(conditional_fisher(X,V[j:j+1],s,cx,Q))
        all_scores, all_F = np.asarray(all_scores), np.asarray(all_F)
        for beta in cfg['betas']:
            w = beta**np.arange(T-1,-1,-1)
            Fh = np.einsum('t,tij->ij',w,all_F)
            Fs = np.einsum('t,tij->ij',w*w,all_F)
            scores = np.einsum('t,tbd->bd',w,all_scores)
            empirical, _, mcse = score_statistics(scores)
            function = lambda z: sum(w[j]*expected_nll(X,z,s,q,cx,
                None if channels[j]=='marginal' else V[j:j+1],Q) for j in range(T))
            H = finite_hessian(function,dimension,.001)
            for i in range(dimension):
                for j in range(dimension):
                    weighted.append(dict(complex_data=cx,n=n,k=k,beta=beta,i=i,j=j,
                        prescribed_channels=','.join(channels),observations=T,episodes=episodes,
                        expected_hessian=Fh[i,j],finite_hessian=H[i,j],expected_score_covariance=Fs[i,j],
                        empirical_score_covariance=empirical[i,j],product_mcse=mcse[i,j]))
    frames = dict(fisher_matrices=matrices,case_checks=checks,finite_differences=finite,
                  likelihood_checks=algebra,curvature=curvature,weighted_mixture=weighted,
                  quadrature_reference=reference)
    for name, rows in frames.items():
        frames[name] = pd.DataFrame(rows)
        frames[name].to_csv(out/(name+'.csv'),index=False)
    grid = np.linspace(-.6,.6,31 if cfg['profile']=='smoke' else 81)
    kl, quadratic, names, slices = [], [], [], []
    ns, ks, cxs = [], [], []
    for cx in (False,True):
        for channel in ('marginal','conditional'):
            n,k = 3,(2 if channel=='conditional' and not cx else 1)
            X,Q = np.eye(n)[:,:k],np.eye(n)[:,k:]
            V = np.array([[1.,0.,0.],[1.,.1,0.],[1.,0.,.1]]) if channel=='conditional' else None
            F = passive_fisher_information(n,k,.7,q,complex_data=cx) if V is None else conditional_fisher(X,V,.7,cx,Q)
            dim = len(F)
            K = np.empty((len(grid),len(grid))); B = np.empty_like(K)
            for iy,y in enumerate(grid):
                for ix,x in enumerate(grid):
                    z=np.zeros(dim);z[:2]=x,y
                    K[iy,ix]=gaussian_kl(X,z,.7,q,cx,V,Q)
                    B[iy,ix]=.5*z@F@z
            kl.append(K);quadratic.append(B);slices.append(F[:2,:2])
            names.append(('complex' if cx else 'real')+'_'+channel)
            ns.append(n);ks.append(k);cxs.append(cx)
    np.savez_compressed(out/'kl_grid.npz',x=grid,y=grid,names=names,kl=kl,
        fisher_quadratic=quadratic,fisher=slices,n=ns,k=ks,complex_data=cxs)
    return frames


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['smoke','paper'],default='smoke')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();cfg=configuration(args.profile)
    out=args.output or ROOT/'results/implementation_audit'/args.profile/'information'
    out.mkdir(parents=True,exist_ok=True)
    sources={name:sha(ROOT/name) for name in SOURCES}
    declaration=dict(config=cfg,sources=sources,source_commit=cfg['source_commit'])
    prior=out/'manifest.json'
    if prior.exists():
        meta=json.loads(prior.read_text())
        if meta['sources']!=sources or meta['config']!=cfg:
            raise RuntimeError('Preserve the previous study and use a fresh output directory')
        for name,value in meta['tables'].items():
            if sha(out/name)!=value:raise RuntimeError('Changed information artifact: '+name)
        print('Verified cached information study');return
    if args.profile=='paper':
        from lib.implementation_validation import require_gate
        require_gate(ROOT)
    (out/'preregistered.json').write_text(json.dumps(declaration,indent=2)+'\n')
    start=perf_counter();frames=run(cfg,out)
    if sources!={name:sha(ROOT/name) for name in SOURCES}:raise RuntimeError('Sources changed during information generation')
    last=frames['finite_differences'];last=last[last.step.eq(min(cfg['finite_difference_steps']))]
    scale=np.maximum(abs(last.fisher),1.)
    fd_error=float(np.max(abs(last.negative_expected_hessian-last.fisher)/scale))
    kl_error=float(np.max(abs(last.kl_hessian-last.fisher)/scale))
    ref=frames['quadrature_reference'];ref=ref[ref.order.eq(max(cfg['quadrature_orders']))]
    checks=dict(algebra_failures=int(frames['likelihood_checks'].status.ne('passed').sum()),
        unresolved_score_cases=int(frames['case_checks'].status.ne('passed').sum()),
        final_fd_relative_error=fd_error,final_kl_relative_error=kl_error,
        maximum_native_reference_mean_error=float(ref.mean_error.max()),
        maximum_native_reference_logz_error=float(ref.logz_error.max()),
        maximum_reference_resolution=float(max(ref.mean_resolution.max(),ref.logz_resolution.max())))
    numerical_ok=bool(checks['algebra_failures']==0 and fd_error<cfg['thresholds']['final_fd_relative_error']
        and kl_error<cfg['thresholds']['final_fd_relative_error']
        and checks['maximum_native_reference_mean_error']<cfg['thresholds']['native_reference']
        and checks['maximum_native_reference_logz_error']<cfg['thresholds']['native_reference'])
    meta=dict(**declaration,status='completed',checks=checks,numerical_ok=numerical_ok,
        seconds=perf_counter()-start,created_utc=datetime.now(timezone.utc).isoformat(),
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        tables={p.name:sha(p) for p in sorted(out.iterdir()) if p.suffix in ('.csv','.npz')},
        scope='Deterministic mathematical references and independent score simulations; not tracking performance, an estimator CRB, or posterior-normalizer Hessian validation.')
    (out/'manifest.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
    print(json.dumps(checks,indent=2),flush=True)
    if not numerical_ok:raise RuntimeError('Information algebra/reference checks failed; inspect retained diagnostics')


if __name__=='__main__':
    main()

"""Independent references and multi-start stationary-block EM experiments."""
from pathlib import Path
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import json
import sys
from time import perf_counter
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp
from lib.block_em import BlockData, block_posterior, fit_block_em, expected_residual

SOURCES = ['lib/block_em.py', 'scripts/run_block_em_study.py', 'tests/test_block_em.py',
    'lib/models.py', 'lib/geometry.py', 'lib/spectral_series.py', 'lib/contact_posterior.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def portable(value):
    if isinstance(value, dict):
        return {str(k): portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(v) for v in value]
    if isinstance(value, np.generic):
        return portable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save(path, value):
    path.write_text(json.dumps(portable(value), indent=2, allow_nan=False)+'\n')


def configuration(profile):
    return dict(profile=profile, source_commit='6e78cc9911e7e78452633f13dd947afa669c3470',
        model='One constant subspace and one fixed boundary prior per disjoint block',
        variance_convention='Real per-coordinate variance; proper-complex covariance, not real/imaginary variance',
        seeds=list(range(915130, 915133)) if profile == 'smoke' else list(range(925130, 925154)),
        block_length=12 if profile == 'smoke' else 32, n=3,
        iterations=12 if profile == 'smoke' else 48, floor=1e-6,
        moment_tolerance=1e-10, max_order=4096, ascent_tolerance=1e-8,
        reference_orders=[40, 64] if profile == 'smoke' else [64, 96],
        real_signal_variance=.9, real_sensor_variance=.3,
        conditional_sensor_variance=.04, input_scale=.6,
        models=['real_marginal_rank_one', 'real_marginal_rank_two',
                'complex_marginal_rank_one', 'conditional_contact_plane'],
        marginal_starts=[dict(name='1', s=.12, tau2=.15), dict(name='2', s=.6, tau2=.7),
                         dict(name='3', s=1.2, tau2=.25), dict(name='0', s=.6, tau2=0.)],
        conditional_starts=[dict(name='1', s=.02), dict(name='2', s=.2), dict(name='3', s=.6)],
        selection='Largest normalized block evidence among positive-signal starts; zero boundary is a declared initialization diagnostic',
        performance_scope='Finite disjoint-block noise estimates, not online adaptation, theorem-scale estimation or global EM convergence',
        guide_source='Uploaded implementation_prompt.md; named main.tex/supplementary.tex unavailable in checkout')


def generated_block(model, seed, cfg):
    complex_data = model.startswith('complex')
    conditional = model.startswith('conditional')
    k = 2 if model in ('real_marginal_rank_two', 'conditional_contact_plane') else 1
    rng = np.random.default_rng(seed)
    n, L = cfg['n'], cfg['block_length']
    X = rng.normal(size=(n, n))
    if complex_data:
        X = X+1j*rng.normal(size=X.shape)
    X = np.linalg.qr(X)[0][:, :k]
    P = X@X.conj().T
    s = cfg['conditional_sensor_variance'] if conditional else cfg['real_sensor_variance']
    noise = rng.normal(size=(L, n))
    if complex_data:
        noise = (noise+1j*rng.normal(size=noise.shape))/np.sqrt(2)
    if conditional:
        V = cfg['input_scale']*rng.normal(size=(L, n))
        U = V@P.T + np.sqrt(s)*noise
        return BlockData(U, k, V), s, None
    coefficient = rng.normal(size=(L, k))
    if complex_data:
        coefficient = (coefficient+1j*rng.normal(size=coefficient.shape))/np.sqrt(2)
    signal = cfg['real_signal_variance']
    U = np.sqrt(signal)*coefficient@X.T+np.sqrt(s)*noise
    return BlockData(U, k, complex_data=complex_data), s, signal


def reference_integral(data, s, tau2, order):
    """Independent Gaussian-product integration, real S² or native complex CP².

    Complex squared eigen-coordinates are Dirichlet(1,1,1), with simplex
    Jacobian 2(1-r).  After diagonalizing the *summed* exponent its phase
    integral is analytic; it is never replaced by a real sphere measure.
    """
    n = data.n
    if n != 3:
        raise ValueError('Reference uses N=3')
    t, w = leggauss(order)
    if not data.complex_data:
        phi = 2*np.pi*(np.arange(2*order)+.5)/(2*order)
        vectors = np.stack(np.broadcast_arrays(np.sqrt(1-t[:, None]**2)*np.cos(phi),
            np.sqrt(1-t[:, None]**2)*np.sin(phi), t[:, None]), axis=-1).reshape(-1, 3)
        weights = np.broadcast_to(w[:, None]/(4*order), (order, 2*order)).ravel()
        P = vectors[:, :, None]*vectors[:, None, :]
        if data.k == 2:
            P = np.eye(3)-P
        prior_energy = np.einsum('ij,bji->b', data.boundary_prior, P).real
        denominator = logsumexp(np.log(weights)+prior_energy)
        numerator = np.log(weights)+prior_energy
        factor = .5
    else:
        # Reconstruct the complete exponent independently from the Gaussian
        # inverse-covariance identity; no call to lib.models.increment.
        S = data.U.T@data.U.conj()
        A = data.boundary_prior + tau2/(s*(s+tau2))*S
        _, basis = np.linalg.eigh(A)
        r, weights1 = (t+1)/2, w/2
        r1, r2 = np.meshgrid(r, r, indexing='ij')
        squared = np.stack([r1, (1-r1)*r2, (1-r1)*(1-r2)], axis=-1).reshape(-1, 3)
        weights = (2*(1-r1)*weights1[:, None]*weights1[None, :]).ravel()
        vectors = np.sqrt(squared)@basis.T
        P = vectors[:, :, None]*vectors[:, None, :].conj()
        numerator = np.log(weights)+np.einsum('ij,bji->b', data.boundary_prior, P).real
        eig0 = np.linalg.eigvalsh(data.boundary_prior)
        denominator = logsumexp(np.log(weights)+squared@eig0)
        factor = 1.
    for j, u in enumerate(data.U):
        if data.V is None:
            R = s*np.eye(n)+tau2*P
            residual = np.broadcast_to(u, (len(P), n))
        else:
            R = np.broadcast_to(s*np.eye(n), (len(P), n, n))
            residual = u-P@data.V[j]
        exponent = np.einsum('bi,bij,bj->b', residual.conj(), np.linalg.inv(R), residual).real
        numerator += -factor*(n*np.log(np.pi if data.complex_data else 2*np.pi)
                              + np.linalg.slogdet(R)[1] + exponent)
    total = logsumexp(numerator)
    posterior_weights = np.exp(numerator-total)
    if data.complex_data:
        # Uniform independent phases remove off-diagonal posterior moments.
        M = (basis*(posterior_weights@squared))@basis.conj().T
    else:
        M = np.einsum('b,bij->ij', posterior_weights, P)
    return float(total-denominator), M


def reference_checks(cfg):
    rows = []
    for index, model in enumerate(cfg['models']):
        local = dict(cfg, block_length=5)
        data, s, signal = generated_block(model, 951300+index, local)
        A0 = np.array([[.4, .1, .05], [.1, -.2, .0], [.05, .0, .1]])
        data = BlockData(data.U*.5, data.k, data.V, data.complex_data, A0)
        result = block_posterior(data, s, signal)
        lo_e, lo_m = reference_integral(data, s, signal, cfg['reference_orders'][0])
        hi_e, hi_m = reference_integral(data, s, signal, cfg['reference_orders'][1])
        evidence_error, mean_error = abs(hi_e-result.log_evidence), np.linalg.norm(hi_m-result.mean, 2)
        resolution_error = max(abs(hi_e-lo_e), np.linalg.norm(hi_m-lo_m, 2))
        rows.append(dict(model=model, n=data.n, k=data.k, complex_data=data.complex_data,
            order_low=cfg['reference_orders'][0], order_high=cfg['reference_orders'][1],
            native_log_evidence=result.log_evidence, reference_log_evidence=hi_e,
            evidence_error=evidence_error, mean_error=mean_error,
            resolution_error=resolution_error, tolerance=2e-8,
            numerical_ok=bool(result.numerical_ok and max(evidence_error, mean_error, resolution_error)<2e-8)))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['smoke', 'paper'], default='smoke')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.profile == 'paper':
        from lib.implementation_validation import require_gate
        require_gate(ROOT)
    cfg = configuration(args.profile)
    out = args.output or ROOT/'results/implementation_audit'/args.profile/'em'
    sources = {name: sha(ROOT/name) for name in SOURCES}
    if (out/'manifest.json').exists():
        old = json.loads((out/'manifest.json').read_text())
        if old['sources'] != sources or old['config'] != cfg:
            raise RuntimeError('Preserve this result and regenerate in a fresh directory')
        for name, expected in old['tables'].items():
            if sha(out/name) != expected:
                raise RuntimeError('Changed completed output: '+name)
        print('Verified completed', args.profile, 'EM component')
        return
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Preserve the incomplete output and use a fresh directory')
    out.mkdir(parents=True, exist_ok=True)
    save(out/'preregistered.json', dict(config=cfg, sources=sources,
        created_utc=datetime.now(timezone.utc).isoformat()))
    begun = perf_counter()
    references = reference_checks(cfg)
    references.to_csv(out/'references.csv', index=False)
    if not references.numerical_ok.all():
        raise ArithmeticError('Independent block reference failed; no Monte Carlo runs launched')
    fits, histories, failures, residuals = [], [], [], []
    for model_index, model in enumerate(cfg['models']):
        for trial, base_seed in enumerate(cfg['seeds']):
            seed = base_seed+1000*model_index
            data, true_s, true_signal = generated_block(model, seed, cfg)
            starts = cfg['conditional_starts'] if data.conditional else cfg['marginal_starts']
            episode_rows = []
            for start in starts:
                try:
                    fitted = fit_block_em(data, start['s'], start.get('tau2'),
                        iterations=cfg['iterations'], floor=cfg['floor'],
                        tol=cfg['moment_tolerance'], max_order=cfg['max_order'],
                        ascent_tolerance=cfg['ascent_tolerance'])
                except ArithmeticError as error:
                    failures.append(dict(model=model, trial=trial, seed=seed, start=start['name'],
                                         numerical_ok=False, reason=str(error)))
                    continue
                histories.extend(dict(model=model, trial=trial, seed=seed, start=start['name'], **h)
                                 for h in fitted.history)
                row = dict(model=model, trial=trial, seed=seed, n=data.n, k=data.k,
                    start=start['name'], true_s=true_s, true_tau2=true_signal,
                    s=fitted.s, tau2=fitted.tau2, log_evidence=fitted.objectives[-1],
                    evidence_gain=fitted.objectives[-1]-fitted.objectives[0],
                    accepted=fitted.accepted, rejected=fitted.rejected, converged=fitted.converged,
                    min_evidence_increment=min(np.diff(fitted.objectives), default=0.),
                    numerical_ok=fitted.posterior.numerical_ok and fitted.rejected == 0,
                    selected=False, absorbing_boundary=start['name']=='0')
                episode_rows.append(row)
                if data.conditional:
                    expected = expected_residual(data, fitted.posterior.mean)
                    mean_only = np.linalg.norm(data.U-data.V@fitted.posterior.mean.T)**2
                    residuals.append(dict(model=model, trial=trial, start=start['name'],
                        expected_residual=expected, mean_only_residual=mean_only,
                        uncertainty_term=expected-mean_only,
                        fitted_s=fitted.s, expected_update=expected/(data.n*len(data.U)),
                        mean_only_update=mean_only/(data.n*len(data.U))))
            eligible = [r for r in episode_rows if not r['absorbing_boundary'] and r['numerical_ok']]
            if eligible:
                max(eligible, key=lambda r: r['log_evidence'])['selected'] = True
            fits.extend(episode_rows)
        print(model, 'completed', len(cfg['seeds']), 'independent blocks', flush=True)
    frames = dict(fits=pd.DataFrame(fits), iterations=pd.DataFrame(histories),
                  failures=pd.DataFrame(failures, columns=['model', 'trial', 'seed', 'start', 'numerical_ok', 'reason']),
                  residuals=pd.DataFrame(residuals), references=references)
    for name, frame in frames.items():
        frame.to_csv(out/(name+'.csv'), index=False)
    selected = frames['fits'][frames['fits'].selected]
    checks = dict(independent_blocks=len(selected), fits=len(fits), numerical_failures=len(failures),
        rejected_steps=int(frames['fits'].rejected.sum()),
        maximum_reference_evidence_error=float(references.evidence_error.max()),
        maximum_reference_mean_error=float(references.mean_error.max()),
        minimum_evidence_increment=float(frames['fits'].min_evidence_increment.min()),
        boundary_signal_maximum=float(frames['fits'].loc[frames['fits'].absorbing_boundary, 'tau2'].abs().max()),
        converged_fits=int(frames['fits'].converged.sum()),
        seconds=perf_counter()-begun,
        fixed_boundary_prior=True, evidence_guard=True,
        scope='Numerical ascent under one stationary-block objective; finite iterations and scale recovery do not prove global/online convergence')
    if any(sha(ROOT/name) != expected for name, expected in sources.items()):
        raise RuntimeError('Producer source changed during the run')
    tables = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    save(out/'manifest.json', dict(profile=args.profile, status='completed', config=cfg,
        sources=sources, tables=tables, checks=checks))
    print(json.dumps(portable(checks), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

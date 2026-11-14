from time import perf_counter
import numpy as np
from scipy.optimize import brentq
from .geometry import center
from .beamforming_protocol import proper_complex, haar_line, ula
from .beamforming_numerics import complex_moments
from .beamforming_decisions import isotropized_mean
from .beamforming_baselines import distortionless_weight, covariance_shrinkage
from benchmarks.common.great import GREAT
from benchmarks.case1.baselines import sample_output_power
SOURCE_COMMIT='ec998a933fd693c2f5bdb62c6bf5654e9345853a'
CONDITIONS=('static','smooth','abrupt','noise_change','power_change')
METHODS=('mean','mode','isotropic','great_window','covariance_loading')



def matched_overlap_angle(n, overlap=.5):
    """First-lobe ULA angle with a declared squared desired overlap."""
    if n < 2 or not 0 < overlap < 1:
        raise ValueError('Require n>=2 and overlap strictly between zero and one')
    desired = ula(n, 0.)
    return float(brentq(lambda theta: abs(np.vdot(desired, ula(n, theta)))**2-overlap,
                        0., np.arcsin(min(1., 2./n)), xtol=1e-14))


def generate_stream(n, length, condition, seed, sigma2=.2, tau2=1., overlap=.5, dt=.01):
    """Generate interference-only snapshots and separate evaluation records."""
    if condition not in CONDITIONS or n < 2 or length < 1:
        raise ValueError('Unknown condition or invalid dimension/length')
    if sigma2 <= 0 or tau2 <= 0 or dt <= 0:
        raise ValueError('Positive variances and sampling interval required')
    children = np.random.SeedSequence(seed).spawn(3)
    rng = np.random.default_rng(children[0])
    theta = matched_overlap_angle(n, overlap)
    angles = np.full(length, theta)
    if condition == 'smooth':
        angles = np.linspace(theta, 3*theta, length)
    elif condition == 'abrupt':
        angles[length//2:] = 3*theta
    sigma = np.full(length, sigma2)
    power = np.full(length, tau2)
    if condition == 'noise_change':
        sigma[length//2:] *= 4.
    elif condition == 'power_change':
        power[length//2:] *= 4.
    h = np.array([ula(n, value) for value in angles])
    z = proper_complex(rng, length)*np.sqrt(power)
    e = proper_complex(rng, (length, n))*np.sqrt(sigma)[:, None]
    U = h*z[:, None] + e
    initial = haar_line(np.random.default_rng(children[1]), n)
    return dict(U=U, h=h, a=ula(n, 0.), sigma2=sigma, tau2=power,
        angles=angles, initial=initial, time=np.arange(1, length+1)*dt,
        heldout_seed=children[2], condition=condition, baseline_angle=theta,
        estimator_inputs='U, stated variances and desired steering for decisions; h and true covariance are scoring only')


def beam_episode(cfg, n, condition, trial, split='test', stationary=False, length=None):
    if split not in ('validation', 'test'):
        raise ValueError('Use a disjoint validation or test split')
    seed = cfg['validation_seed'] if split == 'validation' else cfg['test_seed']
    stream_seed = np.random.SeedSequence([seed, n, CONDITIONS.index(condition), int(trial)])
    # SeedSequence is accepted by generate_stream through a reproducible state.
    seed_words = stream_seed.generate_state(4).tolist()
    length = (cfg['stationary_T'] if stationary else cfg['moving_T']) if length is None else int(length)
    return generate_stream(n, length, condition, seed_words, cfg['sigma2'],
                           cfg['tau2'], cfg['overlap'], cfg['dt'])


def heldout_snapshots(stream, t, draws):
    """Independent fresh disturbances at the present checkpoint."""
    parent = stream['heldout_seed']
    words = parent.generate_state(4).tolist()
    rng = np.random.default_rng(np.random.SeedSequence(words+[int(t)]))
    return (proper_complex(rng, draws, stream['tau2'][t-1])[:, None]*stream['h'][t-1]
            + proper_complex(rng, (draws, stream['U'].shape[1]), stream['sigma2'][t-1]))


def projector_mode(A):
    values, E = np.linalg.eigh(A)
    h = E[:, -1]
    gap = float(values[-1]-values[-2])
    return np.outer(h, h.conj()), h, gap


def modal_weight(h, a, sigma2, tau2, rho=1.):
    """Efficient rank-one modal covariance action, including infinite loading."""
    if not 0 <= rho <= 1:
        raise ValueError('rho must be in [0,1]')
    n = len(a)
    c = sigma2+tau2*(1-rho)/n
    shrink = tau2*rho/(c+tau2*rho)
    overlap = np.vdot(h, a)
    raw = a-shrink*h*overlap
    denominator = np.vdot(a, raw).real
    if denominator <= 0:
        raise ArithmeticError('Invalid positive-definite modal decision')
    w = raw/denominator
    return w/np.conj(np.vdot(w, a))


def covariance_power(w, h, sigma2, tau2):
    return float(sigma2*np.vdot(w, w).real+tau2*abs(np.vdot(w, h))**2)


def checked_moments(A,tol=1e-10,max_order=4096,fallback_max_order=8192):
    """Declared cap extension without relaxing the numerical tolerance.

    The legacy positive-series backend chooses its safe arithmetic branch from
    the cap. Trying its regular cap first avoids unnecessarily using logarithmic
    convolution for moderate n=128 states that already resolve below that cap.
    """
    try:
        result=complex_moments(A,tol,max_order)
        return result,False
    except ArithmeticError:
        if fallback_max_order<=max_order:
            raise
        return complex_moments(A,tol,fallback_max_order),True


def action_from_parameter(A,a,sigma2,tau2,method='mean',tol=1e-10,max_order=4096,
                          fallback_max_order=8192):
    """One production action for cadence timing, without any scoring work.

    Native mean evaluation already includes its eigensystem. A mode-only
    action uses the efficient rank-one inverse identity. Optional uncertainty
    reports and variance fitting are separate operations.
    """
    if method=='mode':
        P,h,gap=projector_mode(A)
        return dict(weight=modal_weight(h,a,sigma2,tau2),projector=P,
                    numerical_ok=True,mode_gap=gap)
    if method not in ('mean','isotropic'):
        raise ValueError('Use mean, mode or isotropic')
    posterior,fallback=checked_moments(A,tol,max_order,fallback_max_order)
    M=posterior.mean
    if method=='isotropic':
        P,h,gap=projector_mode(A)
        M,_=isotropized_mean(M,P)
    return dict(weight=distortionless_weight(sigma2*np.eye(len(A))+tau2*M,a),
                projector=M,numerical_ok=posterior.numerical_ok,
                moment_diagnostics=dict(posterior.diagnostics,cap_extension=fallback),
                log_normalizer=posterior.log_normalizer)


def oracle_power(a, h, sigma2, tau2):
    return covariance_power(modal_weight(h, a, sigma2, tau2), h, sigma2, tau2)


def weighted_statistics(U, sigma2, tau2, beta, times):
    """One assimilation per observation; unnormalized likelihood statistics."""
    n = U.shape[1]
    S = np.zeros((n, n), complex)
    A = np.zeros_like(S)
    mass = 0.
    output = {}
    for t, u in enumerate(U, 1):
        outer = np.outer(u, u.conj())
        s, q = float(sigma2[t-1]), float(tau2[t-1])
        S = beta*S+outer
        A = beta*A+q/(s*(s+q))*outer
        mass = beta*mass+1.
        if t in times:
            output[t] = dict(A=center(A), S=S.copy(), mass=mass)
    return output


def decision_records(U, sigma2, tau2, a, initial, policy, cfg, return_states=False):
    n=U.shape[1];times=policy['times'];beta=policy['shared_beta']
    betas={beta,*[item['covariance_beta'] for item in policy['policies']]}
    snapshots={b:weighted_statistics(U,sigma2,tau2,b,times) for b in betas}
    tracker=GREAT(n,1,complex_data=True,initial_basis=initial[:,None],
        **{key:policy['great_window'][key] for key in ('window','beta','stepsize','inner_steps')})
    great={}
    for t,u in enumerate(U,1):
        X=tracker.update(u)
        if t in times:great[t]=X[:,0].copy()
    records=[]
    for t in times:
        selected=next(item for item in policy['policies'] if t in item['times'])
        state=snapshots[beta][t];P,h,gap=projector_mode(state['A'])
        s,q=float(sigma2[t-1]),float(tau2[t-1])
        posterior,fallback=checked_moments(state['A'],cfg['moment_tolerance'],
            cfg['moment_max_order'],cfg['moment_fallback_max_order'])
        if not posterior.numerical_ok:raise ArithmeticError('Unresolved posterior moment')
        M=posterior.mean;Mi,rho=isotropized_mean(M,P)
        Sstate=snapshots[selected['covariance_beta']][t]
        Rloaded=covariance_shrinkage(Sstate['S']/Sstate['mass'],selected['alpha'])
        weights=dict(mode=modal_weight(h,a,s,q),
            covariance_loading=distortionless_weight(Rloaded,a),
            great_window=modal_weight(great[t],a,s,q),
            mean=distortionless_weight(s*np.eye(n)+q*M,a),
            isotropic=distortionless_weight(s*np.eye(n)+q*Mi,a))
        record=dict(t=t,weights=weights,mean_ok=True,mean_diagnostics=dict(posterior.diagnostics,
            cap_extension=fallback),policy=selected,mode_gap=gap)
        if return_states:record.update(A=state['A'],M=M,P=P,M_iso=Mi,
            log_normalizer=posterior.log_normalizer)
        records.append(record)
    return records


def evaluate_episode(cfg, policy, trial, stationary=False, save_story=False):
    n,condition=policy['n'],policy['condition']
    data=beam_episode(cfg,n,condition,trial,'test',stationary)
    estimated=decision_records(data['U'],data['sigma2'],data['tau2'],data['a'],
        data['initial'],policy,cfg,save_story)
    methods=('mean','mode','great_window','covariance_loading') if stationary else ('mean','mode','isotropic','great_window')
    rows=[];states={}
    for record in estimated:
        t=record['t'];s,q,h=data['sigma2'][t-1],data['tau2'][t-1],data['h'][t-1]
        oracle=oracle_power(data['a'],h,s,q)
        for method in methods:
            w=record['weights'][method];power=covariance_power(w,h,s,q)
            if not np.isfinite(power):raise ArithmeticError('Unresolved beamformer')
            rows.append(dict(n=n,condition=condition,trial=trial,t=t,method=method,
                beta=record['policy']['covariance_beta'] if method=='covariance_loading' else policy['shared_beta'],
                true_power=power,oracle_power=oracle,true_ratio_db=float(10*np.log10(power/oracle)),
                numerical_ok=True,mean_ok=record['mean_ok']))
        if save_story:
            for key in ('A','M','P','M_iso'):states[f'{condition}_n{n}_t{t}_{key}']=record[key]
    return rows,states

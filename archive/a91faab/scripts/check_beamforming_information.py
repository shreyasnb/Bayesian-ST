"""Independent Gaussian-score and physical-angle checks for passive arrays."""
from pathlib import Path
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'1')
import argparse
import hashlib
import json
import platform
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import scipy
from scipy.linalg import null_space
from lib.beamforming_protocol import output_directory, profile as study_profile, proper_complex
from lib.beamforming_theory import (passive_covariance, passive_fisher_information,
    passive_scores, ula_angle_information, ula_signature)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def sources():
    return {name:sha(ROOT/name) for name in ['lib/beamforming_theory.py',
        'lib/beamforming_protocol.py','scripts/check_beamforming_information.py',
        'lib/models.py','lib/geometry.py']}


def expected_likelihood_curvature(X,s,q,step):
    """Finite differences of an independently evaluated expected Gaussian NLL.

    The candidate covariance is formed from a retracted complex basis. Its
    Gaussian determinant and inverse are evaluated directly; no analytic
    Fisher formula or finite Bingham-normalizer Hessian enters this check.
    """
    n,k=X.shape;Q=null_space(X.conj().T);m=k*(n-k);R0=passive_covariance(X@X.conj().T,s,q)
    def objective(coordinate):
        Z=(coordinate[:m]+1j*coordinate[m:]).reshape((n-k,k),order='F')
        Y=np.linalg.qr(X+Q@Z)[0]
        R=s*np.eye(n)+q*(Y@Y.conj().T)
        return float(np.linalg.slogdet(R)[1]+np.trace(np.linalg.solve(R,R0)).real)
    d=2*m;zero=np.zeros(d);base=objective(zero);H=np.empty((d,d));unit=step*np.eye(d)
    for i in range(d):
        H[i,i]=(objective(unit[i])-2*base+objective(-unit[i]))/step**2
        for j in range(i):
            H[i,j]=H[j,i]=(objective(unit[i]+unit[j])-objective(unit[i]-unit[j])
                -objective(-unit[i]+unit[j])+objective(-unit[i]-unit[j]))/(4*step**2)
    return H


def score_covariance(X,s,q,draws,rng,batch=8192):
    n,k=X.shape;d=2*k*(n-k);sum_score=np.zeros(d);cross=np.zeros((d,d));squared_cross=np.zeros((d,d))
    for first in range(0,draws,batch):
        count=min(batch,draws-first)
        # z is a hidden Gaussian coefficient. It is used only to generate
        # observations, not substituted as a known-input measurement channel.
        U=proper_complex(rng,(count,k),q)@X.T+proper_complex(rng,(count,n),s)
        score=passive_scores(U,X,s,q)
        sum_score+=score.sum(axis=0);cross+=score.T@score
        squared=score*score;squared_cross+=squared.T@squared
    mean=sum_score/draws
    covariance=(cross-draws*np.outer(mean,mean))/(draws-1)
    products=cross/draws
    # Monte Carlo SE of uncentered score-product means. Centering changes this
    # diagnostic at smaller order; it is not an exact finite-sample enclosure.
    mcse=np.sqrt(np.maximum(0,squared_cross/draws-products*products)/draws)
    return covariance,mean,mcse


def run(config,out):
    matrix_rows=[];angle_rows=[];case_rows=[];states=[];rng=np.random.default_rng(config['state_seed'])
    n=config['n'];haar=proper_complex(rng,n);haar/=np.linalg.norm(haar)
    for state_id,(name,h) in enumerate([('Haar fixed',haar),('ULA 30 deg',ula_signature(n,np.pi/6))]):
        states.append(h);X=h[:,None]
        for s in config['noises']:
            q=config['q'];seed=np.random.SeedSequence([config['seed'],config['profile_code'],state_id,int(round(1000*s))])
            empirical,mean,mcse=score_covariance(X,s,q,config['draws'],np.random.default_rng(seed),config['batch'])
            analytic=passive_fisher_information(n,1,s,q)
            curvature=expected_likelihood_curvature(X,s,q,config['curvature_step'])
            scale=float(analytic[0,0]);standardized=(empirical-analytic)/np.maximum(mcse,np.finfo(float).tiny)
            mean_z=mean/np.sqrt(np.diag(analytic)/config['draws'])
            for i in range(len(mean)):
                for j in range(len(mean)):
                    matrix_rows.append(dict(profile=config['profile'],n=n,k=1,s=s,q=q,state=name,i=i,j=j,
                        analytic=float(analytic[i,j]),empirical=float(empirical[i,j]),curvature=float(curvature[i,j]),
                        score_product_mcse=float(mcse[i,j]),standardized_residual=float(standardized[i,j]),
                        score_mean_i=float(mean[i]),score_mean_j=float(mean[j]),repeats=config['draws']))
            off=~np.eye(len(mean),dtype=bool)
            case_rows.append(dict(state=name,n=n,s=s,q=q,repeats=config['draws'],
                relative_matrix_error=float(np.linalg.norm(empirical-analytic,'fro')/np.linalg.norm(analytic,'fro')),
                maximum_normalized_offdiagonal=float(np.max(abs(empirical[off]))/scale),
                maximum_score_product_z=float(np.max(abs(standardized))),maximum_score_mean_z=float(np.max(abs(mean_z))),
                expected_curvature_relative_error=float(np.max(abs(curvature-analytic))/scale)))
            if state_id==1:
                Q=null_space(X.conj().T)
                hprime=1j*np.pi*np.arange(n)*np.cos(np.pi/6)*h
                tangent=Q.conj().T@hprime
                coordinate=np.r_[tangent.real,tangent.imag]
                angle_mc=float(coordinate@empirical@coordinate)
            else:angle_mc=np.nan
            for theta in np.deg2rad(config['angles_deg']):
                htheta=ula_signature(n,float(theta));step=config['angle_step']
                finite=(ula_signature(n,float(theta+step))-ula_signature(n,float(theta-step)))/(2*step)
                finite-=htheta*np.vdot(htheta,finite)
                derivative_information=2*q*q/(s*(s+q))*float(np.vdot(finite,finite).real)
                # Keep one angle table, with a genuine score-MC point at 30deg.
                if state_id==1:
                    angle_rows.append(dict(theta_rad=float(theta),theta_deg=float(np.rad2deg(theta)),n=n,s=s,q=q,
                        observations=1,analytic=ula_angle_information(n,float(theta),s,q),
                        finite_derivative=derivative_information,
                        empirical=angle_mc if abs(theta-np.pi/6)<1e-12 else np.nan,
                        repeats=config['draws'] if abs(theta-np.pi/6)<1e-12 else 0,
                        coordinate='Radian angle; degree information is this value times (pi/180)^2'))
            print('Passive information',name,'s=',s,'done',flush=True)
    np.savez_compressed(out/'states.npz',states=np.asarray(states))
    return dict(information_matrix=pd.DataFrame(matrix_rows),angle_information=pd.DataFrame(angle_rows),
                case_checks=pd.DataFrame(case_rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['quick','paper'],default='quick')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();protocol=study_profile(args.profile)
    out=args.output.resolve() if args.output else output_directory(ROOT,args.profile,'information')
    out.mkdir(parents=True,exist_ok=True)
    config=dict(profile=args.profile,n=protocol['n'],k=1,q=protocol['q'],noises=protocol['noises'],
        seed=6571001,state_seed=6571002,profile_code=0 if args.profile=='quick' else 1,
        draws=32768 if args.profile=='quick' else 131072,batch=8192,
        states=['Fixed Haar line, seeded before observations','ULA arrival 30 degrees'],
        angles_deg=np.linspace(-80,80,81 if args.profile=='quick' else 321).tolist(),
        curvature_step=.001,angle_step=1e-6,
        coordinate_order='Re vec_F Z followed by Im vec_F Z; complex tangent dimension 2*k*(n-k)',
        thresholds=dict(expected_curvature_relative_error=1e-4,maximum_score_product_z=7.,maximum_score_mean_z=7.),
        scope='Measurement Gaussian-score information; not a posterior-normalizer Hessian or a CRB for a possibly biased estimator',
        monte_carlo_scope='Independent snapshots at known preassigned truth for a mathematical score check; not a training channel')
    scientific_sources=sources()
    signature=hashlib.sha256(json.dumps(dict(config=config,sources=scientific_sources),sort_keys=True).encode()).hexdigest()
    cached=out/'checks.json'
    if cached.exists():
        previous=json.loads(cached.read_text())
        if previous['source_hash']!=signature:raise RuntimeError('Preserve existing information checks; choose a new output directory')
        if not all(sha(out/name)==value for name,value in previous['tables'].items()):raise RuntimeError('Information table checksum mismatch')
        print('Verified cached information checks');return
    gate=None
    if args.profile=='paper':
        from lib.beamforming_validation import require_beamforming_gate
        gate=require_beamforming_gate(ROOT)
    write_json(out/'preregistered.json',dict(config=config,sources=scientific_sources,source_hash=signature,gate=gate))
    start=perf_counter();frames=run(config,out)
    for name,frame in frames.items():frame.to_csv(out/f'{name}.csv',index=False)
    if sources()!=scientific_sources:raise RuntimeError('Information source changed during evaluation')
    cases=frames['case_checks'];angles=frames['angle_information']
    passed=bool((cases.expected_curvature_relative_error<config['thresholds']['expected_curvature_relative_error']).all()
        and(cases.maximum_score_product_z<config['thresholds']['maximum_score_product_z']).all()
        and(cases.maximum_score_mean_z<config['thresholds']['maximum_score_mean_z']).all())
    angle_error=float(np.max(abs(angles.analytic-angles.finite_derivative)/np.maximum(angles.analytic,1e-15)))
    metadata=dict(status='passed' if passed and angle_error<1e-7 else 'unresolved',source_hash=signature,
        config=config,sources=scientific_sources,gate=gate,seconds=perf_counter()-start,
        maximum_matrix_relative_error=float(cases.relative_matrix_error.max()),
        maximum_standardized_score_residual=float(cases.maximum_score_product_z.max()),
        maximum_score_mean_z=float(cases.maximum_score_mean_z.max()),
        maximum_expected_curvature_relative_error=float(cases.expected_curvature_relative_error.max()),
        maximum_angle_derivative_relative_error=angle_error,
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        tables={p.name:sha(p) for p in sorted(out.iterdir()) if p.suffix in ['.csv','.npz']},
        note='Score-product standard errors are Monte Carlo diagnostics; finite differentiation and truncation are separate errors')
    write_json(out/'checks.json',metadata);write_json(out/'config.json',metadata)
    print(json.dumps({k:v for k,v in metadata.items() if k not in ['config','sources','gate','tables']},indent=2))


if __name__=='__main__':main()

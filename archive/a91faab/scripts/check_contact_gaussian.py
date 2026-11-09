"""Audit local Gaussian geometry, mapped tails, process approximation and cost."""
from pathlib import Path
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import hashlib,json,platform,sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,scipy
from numpy.polynomial.hermite_e import hermegauss
from lib.contact_gaussian import (gaussian_summary,laplace_summary,tangent_basis,
    measurement,measurement_jacobian,exponential_normal,RecursiveTangentGaussian,
    _MappedGaussian)
from lib.contact_posterior import contact_summary
from lib.contact_process_prediction import kernel_parameters,kappa_from_chordal_rms
from lib.contact_mechanics import generate_contact
from lib.contact_study import query_vectors


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out=ROOT/'results/contact_uncertainty/numerics/gaussian';out.mkdir(parents=True,exist_ok=True)
    n=np.array([1.,2.,3.]);n/=np.linalg.norm(n);E=tangent_basis(n);W=query_vectors(1.)
    rng=np.random.default_rng(6245293)
    P=np.array([[.13,.04],[.04,.03]])
    analytic=_MappedGaussian(n,P,E,128).mean()
    x,w=hermegauss(40);w=w/np.sqrt(2*np.pi)
    z=np.array(np.meshgrid(x,x)).reshape(2,-1).T
    weights=(w[:,None]*w).ravel();tangent=z@np.linalg.cholesky(P).T@E.T
    angles=np.linalg.norm(tangent,axis=1)
    normals=np.cos(angles)[:,None]*n+np.sinc(angles/np.pi)[:,None]*tangent
    reference=np.eye(3)-np.einsum('b,bi,bj->ij',weights,normals,normals)
    mean_error=float(np.linalg.norm(analytic-reference,2));assert mean_error<1e-10
    iid=rng.multivariate_normal(np.zeros(2),P,size=200000)
    empirical_cut=float(np.mean(np.linalg.norm(iid,axis=1)>=np.pi/2))
    cut=_MappedGaussian(n,P,E,256).cut_locus_mass()
    assert abs(empirical_cut-cut)<.001
    J=measurement_jacobian(n,W[0],E);h=1e-6
    finite=np.column_stack([(measurement(exponential_normal(n,h*e),W[0])-
                             measurement(exponential_normal(n,-h*e),W[0]))/(2*h) for e in E.T])
    jacobian_error=float(np.max(abs(finite-J)));assert jacobian_error<1e-9
    rows=[]
    for kappa in [0.,4.,20.,200.,2000.,np.inf]:
        kernel=kernel_parameters(kappa)
        local=_MappedGaussian(np.array([1.,0.,0.]),kernel.angle_variance*np.eye(2),order=128)
        M=local.mean();a_gaussian=1-float(M[0,0]);rho_gaussian=(3*a_gaussian-1)/2
        rows.append(dict(kappa=kappa,kernel_angle_variance=kernel.angle_variance,
            kernel_chordal_rms=kernel.chordal_rms,kernel_rho=kernel.rho,
            mapped_gaussian_rho=rho_gaussian,rho_difference=rho_gaussian-kernel.rho,
            gaussian_cut_locus_mass=local.cut_locus_mass()))
    pd.DataFrame(rows).to_csv(out/'process_approximation.csv',index=False)
    shape=[]
    rotation=np.linalg.qr(rng.normal(size=(3,3)))[0]
    for gaps in [(2.,5.),(80.,200.),(80000.,100000.)]:
        A=rotation@np.diag([0.,*gaps])@rotation.T
        resolved=contact_summary(A,W);local=laplace_summary(A,W)
        shape.append(dict(gap1=gaps[0],gap2=gaps[1],
            mean_operator_difference=float(np.linalg.norm(local['mean']-resolved['mean'],2)),
            maximum_quantile_difference=float(max(np.max(abs(local['response_lower']-resolved['response_lower'])),
                                                  np.max(abs(local['response_upper']-resolved['response_upper'])))),
            local_integration_ok=bool(local['numerics']['integration_numerical_ok']),
            local_branch_ok=bool(local['numerics']['local_approximation_ok']),
            cut_locus_mass=local['numerics']['cut_locus_mass'],
            local_seconds=local['numerics']['integration_seconds'],
            resolved_seconds=resolved['numerics']['integration_seconds']))
    assert shape[-1]['mean_operator_difference']<1e-8
    pd.DataFrame(shape).to_csv(out/'same_parameter_shape.csv',index=False)
    # Development-only physical data; neither final comparison root is used.
    data=generate_contact(6245294,0,'sphere_small')
    filter=RecursiveTangentGaussian(1e-6,kappa_from_chordal_rms(.02),warmup=6)
    start=perf_counter();scores=[]
    for u,v in zip(data['U'],data['V']):
        scores.append(filter.predictive_log_score(u,v));filter.update(u,v)
    runtime=perf_counter()-start
    assert filter.initialized and np.isfinite(scores).all()
    assert np.linalg.eigvalsh(filter.covariance).min()>0
    assert abs(filter.normal@filter.normal-1)<1e-12
    result=filter.summary(query_vectors(.02),true_normal=data['normals'][-1])
    report=dict(status='passed',mean_hermite_reference_error=mean_error,
        jacobian_finite_difference_error=jacobian_error,
        iid_cut_locus_mass=empirical_cut,angular_cut_locus_mass=cut,
        development_seed=6245294,development_observations=600,
        filter_seconds=runtime,timings=filter.timings,final_summary_numerics=result['numerics'],
        nonlocal_update_count=filter.nonlocal_updates,
        initialization='Six coherent process-Bingham corrections from uniform law, followed by same-parameter Laplace initialization; all cost retained',
        process_scope='Exact kernel angular second moment matched; exponential-Gaussian rho generally differs from the kernel rho, particularly for broad increments',
        region_scope='Projective chordal balls under mapped law; local tangent-coordinate Gaussian ellipsoid is not invariant-density Grassmann HPD',
        broad_scope='Cut-locus mass above1% is an explicitly unresolved local approximation; no small variance floor or physical interval clipping',
        timing_scope='Prediction/update/score clocks additive. warmup_seconds is a nested startup subset and must not be added again.',
        posterior_scope='Mapped moments use analytic radial Dawson integral; scalar CDFs use all periodic radial roots and exact Gaussian radial primitives with two angular resolutions',
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        sources={p:sha(ROOT/p) for p in ['lib/contact_gaussian.py','scripts/check_contact_gaussian.py','lib/contact_process_prediction.py','lib/contact_posterior.py','lib/contact_mechanics.py']},
        tables={p.name:sha(p) for p in out.glob('*.csv')})
    (out/'checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()

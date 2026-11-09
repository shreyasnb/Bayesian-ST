"""Independent Gaussian, coordinate and perturbation checks for passive theory."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm, null_space
from scipy.stats import multivariate_normal

from lib.beamforming_theory import (exponential_weights, modal_scanning_bound,
    passive_covariance, passive_covariance_certificate,
    passive_fisher_information, passive_log_likelihood, passive_scores,
    passive_tangent_basis, ula_angle_information, ula_derivative, ula_signature)
from lib.models import log_likelihood
from lib.geometry import gaussian


def basis(rng, n, k):
    return np.linalg.qr(rng.normal(size=(n,k))+1j*rng.normal(size=(n,k)))[0]


def test_passive_density_matches_direct_complex_and_structured_real_density():
    rng=np.random.default_rng(5127);X=basis(rng,5,2);P=X@X.conj().T
    u=gaussian(rng,(5,),True);s,q=.3,1.2;R=passive_covariance(P,s,q)
    direct=-(5*np.log(np.pi)+np.linalg.slogdet(R)[1]+np.vdot(u,np.linalg.solve(R,u)).real)
    assert_allclose(passive_log_likelihood(u,P,s,q),direct,atol=1e-12)
    assert_allclose(log_likelihood(u,P,s,tau2=q,complex_data=True),direct,atol=1e-12)
    # This checks the Gaussian normalization only. The inference state is not
    # replaced by the much larger unrestricted real Grassmann manifold.
    real_cov=.5*np.block([[R.real,-R.imag],[R.imag,R.real]])
    real_density=multivariate_normal.logpdf(np.r_[u.real,u.imag],cov=real_cov)
    assert_allclose(direct,real_density,atol=1e-12)


def test_passive_real_factor_and_input_validation():
    P=np.diag([1.,0.,0.]);u=np.array([.2,-.4,.3]);s,q=.2,1.
    assert_allclose(passive_log_likelihood(u,P,s,q,False),
        multivariate_normal.logpdf(u,cov=s*np.eye(3)+q*P),atol=1e-12)
    assert_allclose(passive_fisher_information(3,1,s,q,complex_data=False),
        q*q/(s*(s+q))*np.eye(2))
    assert_allclose(passive_fisher_information(3,1,s,0),np.zeros((4,4)))
    for args in [(3,0,s,q),(3,3,s,q),(3,1,0,q),(3,1,s,-q)]:
        with pytest.raises(ValueError):passive_fisher_information(*args)
    with pytest.raises(ValueError):ula_angle_information(8,np.nan,s,q)


def test_fisher_is_direct_covariance_trace_in_both_real_coordinate_blocks():
    rng=np.random.default_rng(30215);n,k=5,2;X=basis(rng,n,k);s,q=.4,1.3
    R=passive_covariance(X@X.conj().T,s,q);inverse=np.linalg.inv(R)
    derivatives=q*passive_tangent_basis(X)
    direct=np.array([[np.trace(inverse@d@inverse@e).real for e in derivatives] for d in derivatives])
    assert_allclose(direct,passive_fisher_information(n,k,s,q),atol=2e-13)
    # Gauge changes alter the coordinate basis, not this isotropic information.
    U=basis(rng,n,n-1);Q=np.linalg.qr(np.c_[U,gaussian(rng,(n,),True)])[0]
    transformed=passive_tangent_basis(Q@X)
    transformed_R=Q@R@Q.conj().T;transformed_inverse=np.linalg.inv(transformed_R)
    F=np.array([[q*q*np.trace(transformed_inverse@d@transformed_inverse@e).real
                 for e in transformed] for d in transformed])
    assert_allclose(F,direct,atol=2e-13)


def test_score_is_derivative_of_normalized_likelihood_not_posterior_curvature():
    rng=np.random.default_rng(7863);n=4;X=basis(rng,n,1);Q=null_space(X.conj().T)
    U=gaussian(rng,(7,n),True);s,q=.3,1.4;h=2e-6
    score=passive_scores(U,X,s,q,complement=Q)
    columns=[]
    for imaginary in (False,True):
        for a in range(n-1):
            direction=Q[:,[a]]*(1j if imaginary else 1.)
            plus=X+h*direction;plus/=np.linalg.norm(plus)
            minus=X-h*direction;minus/=np.linalg.norm(minus)
            values=[(passive_log_likelihood(u,plus@plus.conj().T,s,q)
                    -passive_log_likelihood(u,minus@minus.conj().T,s,q))/(2*h) for u in U]
            columns.append(values)
    assert_allclose(score,np.array(columns).T,rtol=2e-7,atol=1e-8)


def test_ula_physical_angle_information_and_degree_conversion():
    n=8;s,q=.2,1.;h=1e-6
    for theta in np.deg2rad([-80.,-20.,0.,30.,75.]):
        v=ula_signature(n,theta)
        finite=(ula_signature(n,theta+h)-ula_signature(n,theta-h))/(2*h)
        assert_allclose(finite,ula_derivative(n,theta),atol=2e-9)
        perpendicular=finite-v*np.vdot(v,finite)
        direct=2*q*q/(s*(s+q))*np.vdot(perpendicular,perpendicular).real
        assert_allclose(ula_angle_information(n,theta,s,q),direct,rtol=2e-9)
        assert_allclose(ula_angle_information(n,theta,s,q,13,True),13*direct*(np.pi/180)**2,rtol=2e-9)
    assert ula_angle_information(n,0,s,q)>ula_angle_information(n,np.deg2rad(75),s,q)


def test_covariance_fixture_is_nonvacuous_but_short_and_drifting_bounds_are_not_clipped():
    signatures=np.broadcast_to(np.eye(4,dtype=complex)[:,0],(4096,4))
    result=passive_covariance_certificate(signatures,.2,1.,delta=.05)
    # Independently written scalar net/Bernstein expression.
    x=8*np.log(9)+np.log(40)
    eta=2.4*(np.sqrt(2*x/4096)+x/4096)
    assert_allclose(result.eta,eta,atol=1e-14)
    assert result.valid and .256<result.eta<.258 and .345<result.bound<.347
    assert_allclose(result.bound,eta/(1-eta),atol=1e-13)
    assert_allclose(result.neff,4096)
    assert_allclose(result.weight_sum,4096)
    assert_allclose(modal_scanning_bound(result),result.bound)
    short=passive_covariance_certificate(signatures[:64],.2,1.)
    assert not short.valid and np.isnan(short.bound) and np.isnan(modal_scanning_bound(short))
    path=signatures.copy();path[:2048]=np.eye(4)[:,1]
    drift=passive_covariance_certificate(path,.2,1.)
    assert_allclose(drift.drift,.5,atol=1e-12)
    assert not drift.valid and np.isnan(drift.bound)


def test_weights_and_transported_drift_use_full_history_and_known_motion():
    rng=np.random.default_rng(991);n,T=4,9;X=basis(rng,n,1)[:,0];Gs=[];Hs=[]
    for _ in range(T):
        W=gaussian(rng,(n,n),True);G=expm(.03*(W-W.conj().T));Gs.append(G)
        X=G@X;Hs.append(X.copy())
    beta=.91;alpha,total=exponential_weights(T,beta)
    direct=beta**np.arange(T-1,-1,-1)
    assert_allclose(alpha,direct/direct.sum());assert_allclose(total,direct.sum())
    result=passive_covariance_certificate(Hs,.2,1.,beta=beta,G=np.array(Gs))
    assert result.drift<1e-12
    for j in range(T):
        transported=result.transports[j]@Hs[j]
        assert_allclose(np.outer(transported,transported.conj()),np.outer(Hs[-1],Hs[-1].conj()),atol=1e-12)
    unnormalized=passive_covariance_certificate(Hs,.2,1.,weights=7*direct,G=np.array(Gs))
    assert_allclose(unnormalized.eta,result.eta)
    assert_allclose(unnormalized.neff,result.neff)
    assert_allclose(unnormalized.weight_sum,7*total)
    with pytest.raises(ValueError):passive_covariance_certificate(Hs,.2,1.,weights=direct,beta=.9)


def test_eigenspace_and_scanning_implications_are_deterministic_on_the_event():
    rng=np.random.default_rng(9102);n,k=7,2;X=basis(rng,n,k);P=X@X.conj().T;s,q=.2,1.
    R=s*np.eye(n)+q*P
    for magnitude in [.01,.1,.3,.49]:
        W=gaussian(rng,(n,n),True);E=(W+W.conj().T)/2;E*=magnitude/np.linalg.norm(E,2)
        Y=np.linalg.eigh(R+E)[1][:,-k:];Phat=Y@Y.conj().T
        dc=np.sqrt(max(0,k-np.linalg.norm(X.conj().T@Y,'fro')**2))
        bound=np.sqrt(min(k,n-k))*magnitude/(q-magnitude)
        assert dc<=bound+1e-12
        scan=gaussian(rng,(40,n),True);scan/=np.linalg.norm(scan,axis=1)[:,None]
        error=abs(np.einsum('bi,ij,bj->b',scan.conj(),q*(Phat-P),scan))
        assert np.max(error)<=q*dc+1e-12


def test_invalid_paths_transports_and_zero_gap_are_rejected():
    H=np.tile(np.array([1.,0.,0.]),(5,1))
    with pytest.raises(ValueError):passive_covariance_certificate(H,.2,0.)
    with pytest.raises(ValueError):passive_covariance_certificate(H*2,.2,1.)
    with pytest.raises(ValueError):passive_covariance_certificate(H,.2,1.,weights=np.zeros(5))
    with pytest.raises(ValueError):passive_covariance_certificate(H,.2,1.,G=np.broadcast_to(1.001*np.eye(3),(5,3,3)))

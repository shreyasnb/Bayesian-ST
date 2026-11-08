"""Independent harmonic, likelihood and projective-measure reference checks."""
import numpy as np
from numpy.testing import assert_allclose
from scipy.special import dawsn
from lib.contact_spherical_reference import (SphericalReference,grid_basis,
    kernel_multipliers,sample_watson,harmonic_basis)
from lib.contact_posterior import contact_mean
from lib.models import increment,constant


def test_even_harmonics_are_orthonormal_for_normalized_projective_measure():
    nodes,w,Y,degrees=grid_basis(8,16,32)
    assert_allclose(w.sum(),1,atol=1e-14)
    assert_allclose(Y.T@(w[:,None]*Y),np.eye(Y.shape[1]),atol=2e-13)
    opposite,_=harmonic_basis(-nodes,8)
    assert_allclose(opposite,Y,atol=2e-13)
    assert (degrees%2==0).all()


def test_watson_kernel_uniform_degree_two_and_deterministic_limits():
    uniform=kernel_multipliers(0.,8)
    assert uniform[0]==1
    assert_allclose([uniform[l] for l in [2,4,6,8]],0.,atol=2e-14)
    for kappa in [4.,20.]:
        root=np.sqrt(kappa);a=1/(2*root*dawsn(root))-1/(2*kappa)
        assert_allclose(kernel_multipliers(kappa,8)[2],(3*a-1)/2,atol=2e-13)
    concentrated=kernel_multipliers(1000.,4)
    assert concentrated[2]>.998 and concentrated[4]>.99


def test_uniform_transition_and_gaussian_correction_match_bingham_likelihood():
    u=np.array([.2,-.3,.7]);v=np.array([1.,0.,0.]);s=.12
    ref=SphericalReference(4.,degree=24)
    score=ref.update(u,v,s);B=increment(u,s,v=v)
    mean,diagnostics=contact_mean(B,strict=True)
    assert_allclose(ref.mean(),mean,atol=3e-10)
    assert_allclose(score,constant(u,2,s,v=v)+diagnostics['log_normalizer'],atol=3e-10)
    # A uniform kernel erases the previous posterior exactly.
    erased=SphericalReference(0.,degree=16)
    erased.update(u,v,s);erased.predict()
    assert_allclose(erased.prior,1.,atol=2e-13)


def test_full_density_prediction_has_the_exact_complement_first_moment():
    ref=SphericalReference(20.,degree=24)
    ref.update(np.array([.2,.8,-.1]),np.array([0.,1.,0.]),.12)
    before=ref.mean();rho=kernel_multipliers(20.,24)[2]
    prior=ref.predict()
    N=np.einsum('n,ni,nj->ij',ref.weights*prior,ref.nodes,ref.nodes)
    assert_allclose(np.eye(3)-N,rho*before+(1-rho)*2*np.eye(3)/3,atol=5e-12)
    assert ref.numerical_ok


def test_independent_watson_sampler_respects_unoriented_transition_moments():
    rng=np.random.default_rng(7315273);m=np.array([1.,2.,-1.]);m/=np.linalg.norm(m)
    kappa=6.;X=np.array([sample_watson(m,kappa,rng) for _ in range(12000)])
    a=1/(2*np.sqrt(kappa)*dawsn(np.sqrt(kappa)))-1/(2*kappa);b=(1-a)/2
    assert_allclose(X.T@X/len(X),a*np.outer(m,m)+b*(np.eye(3)-np.outer(m,m)),atol=.008)
    assert_allclose(np.linalg.norm(X,axis=1),1,atol=3e-15)
    assert abs(X.mean(axis=0)).max()<.025


def test_polar_coordinate_rotation_preserves_even_harmonic_density():
    from lib.contact_spherical_reference import polar_axis_rotation
    rng=np.random.default_rng(7315274)
    nodes=rng.normal(size=(47,3));nodes/=np.linalg.norm(nodes,axis=1)[:,None]
    Y,_=harmonic_basis(nodes,8);coefficients=rng.normal(size=Y.shape[1])
    for component in range(3):
        Q,transform=polar_axis_rotation(8,component)
        rotated,_=harmonic_basis(nodes@Q.T,8)
        assert_allclose(Y@transform@coefficients,rotated@coefficients,atol=2e-13)
        assert_allclose(transform.T@transform,np.eye(len(coefficients)),atol=3e-14)


def test_uniform_diagonal_response_quantiles_have_no_axis_ring_atom():
    from lib.contact_spherical_reference import functional_quantiles,grid_cdf
    ref=SphericalReference(4.,degree=8);W=np.eye(3)
    nodes,w,_,_=grid_basis(8,96,192)
    marginals=ref.diagonal_marginals(96,192)
    quantiles=functional_quantiles(96,192,W,w,[.025,.975],marginals)
    # For uniform n, |n_i|~Uniform(0,1), so F_{1-n_i^2}(y)=1-sqrt(1-y).
    for i in range(3):assert_allclose(quantiles[i,i],[1-.975**2,1-.025**2],atol=2e-4)
    cdfs=grid_cdf(96,192,W,w,np.diag([1-.975**2]*3),marginals).reshape(3,3)
    assert_allclose(np.diag(cdfs),.025,atol=1e-4)

"""Independent scalar identities required by the final simulation guide."""
import numpy as np
import pytest

from lib.final_theory import configuration
from scripts.run_final_theory import scalar_checks


@pytest.mark.parametrize('example',['beamforming','contact'])
def test_scalar_identity_table_records_independent_paths_and_limits(example):
    frame=scalar_checks(configuration('verify'),example)
    failed=frame.loc[frame.status.ne('passed'),['check','error','tolerance']]
    assert failed.empty,failed.to_string(index=False)
    assert {'marginal_denoising_mean','marginal_denoising_covariance',
            'marginal_predictive_covariance','full_response_covariance',
            'response_risk_decomposition','modal_response_excess'}<=set(frame.check)
    if example=='contact':
        assert {'contact_zero_task_penalty','contact_uniform_command',
                'contact_uniform_command_cost','contact_expected_command_cost'}<=set(frame.check)


@pytest.mark.parametrize('complex_data',[False,True])
def test_latent_gaussian_conditioning_reduces_to_projector_denoising(complex_data):
    # Direct block-Gaussian conditioning does not use a Bingham increment or
    # moment routine. Singular latent covariance is allowed; observation
    # covariance is positive definite because sigma2>0.
    rng=np.random.default_rng(42019);n,k=5,2
    Z=rng.normal(size=(n,k))
    if complex_data:Z=Z+1j*rng.normal(size=(n,k))
    X=np.linalg.qr(Z)[0];P=X@X.conj().T
    sigma2,tau2=.37,1.6
    latent=tau2*P;observation=latent+sigma2*np.eye(n)
    gain=latent@np.linalg.inv(observation)
    conditional_cov=latent-gain@latent
    assert np.linalg.norm(gain-tau2/(sigma2+tau2)*P)<1e-12
    assert np.linalg.norm(conditional_cov-sigma2*tau2/(sigma2+tau2)*P)<1e-12


def test_uniform_contact_solution_and_zero_penalty_follow_direct_quadratic():
    from lib.contact_geometry_study import soft_action
    for n in (3,8):
        k=n-1;M=k/n*np.eye(n);c=np.arange(1,n+1)/n
        for lam in (0.,.2,4.):
            H=np.eye(n)+lam*(np.eye(n)-M)
            a=soft_action(M,lam)@c
            assert np.linalg.norm(H@a-c)<1e-12
            assert np.linalg.norm(a-c/(1+lam*(1-k/n)))<1e-12
            actual=np.linalg.norm(a-c)**2+lam*a@(np.eye(n)-M)@a
            expected=lam*(1-k/n)/(1+lam*(1-k/n))*np.dot(c,c)
            assert abs(actual-expected)<1e-12

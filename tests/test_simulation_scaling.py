"""Independent checks for the efficient point-decision timing reference."""
import numpy as np
import pytest
from lib.simulation_scaling import point_mvdr,native_mean


def test_woodbury_point_action_matches_independent_covariance_solve():
    rng=np.random.default_rng(2026110817)
    for complex_data in [False,True]:
        for N in [3,16]:
            h=rng.normal(size=N);a=rng.normal(size=N)
            if complex_data:
                h=h+1j*rng.normal(size=N);a=a+1j*rng.normal(size=N)
            h/=np.linalg.norm(h);a/=np.linalg.norm(a)
            for s,tau2 in [(1.,0.),(.2,1.),(.01,10.)]:
                covariance=s*np.eye(N)+tau2*np.outer(h,h.conj())
                solution=np.linalg.solve(covariance,a)
                direct=solution/np.vdot(a,solution).real
                action=point_mvdr(h,a,s,tau2)
                assert np.linalg.norm(action-direct)<1e-10
                assert abs(np.vdot(action,a)-1)<1e-11


def test_scaling_does_not_replace_general_rank_moments():
    with pytest.raises(NotImplementedError):
        native_mean(np.diag([4.,2.,0.,-2.,-4.]),2)
    with pytest.raises(NotImplementedError):
        native_mean(np.diag([4.,2.,0.,-2.,-4.]).astype(complex),2)

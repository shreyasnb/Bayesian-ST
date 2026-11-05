"""Known-motion controls are coupled exactly, without using hidden-state inputs."""
import numpy as np
from numpy.testing import assert_allclose
from lib.drift_study import paired_data,run_trajectory


def test_rotated_inputs_and_noise_preserve_conditional_protocol():
    X0,data=paired_data(74013,0,T=16)
    base=data[0.]
    for rate,d in data.items():
        for j in range(16):
            R=d['rotations'][j]
            assert_allclose(d['v'][j],R@base['v'][j],atol=1e-12)
            assert_allclose(d['u'][j],R@base['u'][j],atol=1e-12)
            assert_allclose(d['X'][j],R@X0,atol=1e-12)
            assert_allclose(R.T@R,np.eye(8),atol=1e-12)


def test_identity_slow_drift_has_informative_bound():
    config=dict(seed=74013,n=8,k=3,T=128,s=.03,input_norm=2.,slow_rate=.003,
        fast_rate=.025,forgetting=.97,delta=.05,report_times=[128],posterior_trials=0,
        draws=100,burnin=100,max_draws=100,max_burnin=100)
    result=run_trajectory((config,0))
    rows={r['protocol']:r for r in result['bounds']}
    assert rows['identity_slow']['valid']
    assert rows['identity_slow']['bound']<rows['identity_slow']['ceiling']
    assert rows['identity_slow']['chordal_error']<rows['identity_slow']['bound']
    assert not rows['identity_fast']['valid']
    assert result['max_equivariance']<1e-8
    assert_allclose(rows['known_slow']['chordal_error'],rows['static']['chordal_error'],atol=1e-10)

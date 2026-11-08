"""Fair-access, validation-isolation and correction bookkeeping regressions."""
import numpy as np
from lib.application_ablation import candidate_families,data,_new_tracker,_update,noise_ablation,evaluate_candidate
from lib.filter import BinghamFilter
from lib.geometry import projector


def test_loaded_covariance_candidates_are_cartesian_and_old_boundaries_extended():
    families=candidate_families('array');p=families['Loaded covariance']
    betas={r['beta'] for r in p};loads={r['loading'] for r in p}
    assert len(p)==len(betas)*len(loads)
    assert set((r['beta'],r['loading']) for r in p)==set((b,l) for b in betas for l in loads)
    field=candidate_families('field')
    assert min(r['step'] for r in field['GROUSE'])<.01
    assert max(r['window'] for r in field['Rolling PCA'])>48


def test_application_streams_are_independent_and_motion_access_declared():
    config=dict(validation_seed=582031,test_seed=582032,T_array=4,T_field=8)
    for app,n in [('array',16),('field',8)]:
        a=data(config,app,0,True);b=data(config,app,0,False)
        assert not np.allclose(a['U'],b['U'])
        np.testing.assert_array_equal(a['G'],np.eye(n))
        scenario=dict(scenario='Partial motion',mismatch='Matched',rate=.012 if app=='array' else .018,motion='partial',primary=False)
        partial=data(config,app,0,False,scenario)
        assert not np.allclose(partial['G'],np.eye(n))
        np.testing.assert_allclose(partial['G'].conj().T@partial['G'],np.eye(n),atol=1e-12)


def test_instrumented_fixed_noise_update_matches_original_once():
    config=dict(validation_seed=582031,test_seed=582032,T_array=8,T_field=8)
    for app in ('array','field'):
        d=data(config,app,0,False)
        f=_new_tracker('Posterior mean',dict(beta=.85),d,3)
        reference=BinghamFilter(d['n'],d['k'],d['s'],d['tau2'],beta=.85,complex_data=d['complex_data'])
        for u in d['U']:
            update,prediction=_update(f,u,d['G'],True)
            reference.update(u,G=d['G'])
            np.testing.assert_allclose(f.A,reference.A,atol=1e-12)
            np.testing.assert_allclose(f.records[-1].prior,reference.records[-1].prior,atol=1e-12)
            assert update>=0 and prediction>=0
        np.testing.assert_allclose(projector(f.basis),projector(reference.basis),atol=1e-12)


def test_noise_ablation_checks_only_frozen_window_ascent():
    result=noise_ablation(seed=51432,trials=1,T=8,beta=.95)
    learned=result[result.method=='Learned noise']
    assert len(learned)==3 and learned.frozen_ascent_checked.all()
    assert (learned.frozen_objective_change>=-1e-8).all()
    assert (learned.stored_window==8).all()
    assert not result.state_space_ascent_claim.any()


def test_bounded_posterior_candidate_preserves_two_decisions_and_diagnostics():
    config=dict(validation_seed=582031,test_seed=582032,T_array=8,T_field=8,
                draws=40,burnin=20,max_draws=40,max_burnin=20)
    rows=evaluate_candidate(config,'field',0,'Posterior mean',dict(beta=.95),True)
    assert [r['method'] for r in rows]==['Posterior mean','Mode']
    assert all(np.isfinite(r['loss']) and r['loss']>=0 for r in rows)
    assert all('sampling_ok' in r and 'min_ess' in r for r in rows)
    assert rows[0]['moment_seconds']>0 and rows[1]['moment_seconds']==0
    assert all(r['prediction_seconds']>0 and r['update_seconds']>0 for r in rows)

from pathlib import Path
import json
import numpy as np
from numpy.testing import assert_allclose
from lib.contact_mechanics import solve_contact_velocity
from lib.contact_process_prediction import fast_moments
from lib.contact_posterior import normal_quadrature
from lib.final_contact import (generate_contact_trial,bingham_records,
    anisotropy_fixture,response_loss,query_directions,test_episode as contact_episode)

ROOT=Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT/'configs/contact.json').read_text())['config']


def test_dense_contact_trajectory_and_posterior_match_pinned_source():
    cfg=config()
    expected=np.load(Path(__file__).parent/'fixtures/audited_contact.npz')
    times=np.array([4,16,64,128,192]);obs=np.array([0,1,3,63,127,143,191])
    for trial in [0,7]:
        data=generate_contact_trial(cfg,'sphere',trial)
        records,A=bingham_records(cfg,data,.9,times)
        for name in ['U','V','G','P','world_normal']:
            assert_allclose(data[name][obs],expected[f'trial{trial}_{name}'],rtol=2e-12,atol=2e-13)
        assert_allclose(A[times-1],expected[f'trial{trial}_A'],rtol=2e-12,atol=2e-12)
        for name in ['M','Pmode','isotropic']:
            assert_allclose([records[t][name] for t in times],expected[f'trial{trial}_{name}'],atol=2e-11)
            assert all(records[t]['diagnostics']['mean_resolution_error']<5e-8 for t in times)


def test_contact_prediction_losses_match_pinned_source():
    cfg=config();data=generate_contact_trial(cfg,'sphere',7)
    times=[4,16,64,128,192]
    matched,_=bingham_records(cfg,data,.9,times)
    unaligned,_=bingham_records(cfg,data,.9,times,transport=False)
    original=json.loads((Path(__file__).parent/'fixtures/audited_contact_metrics.json').read_text())
    produced=contact_episode(cfg,'sphere',7,[{'condition':'sphere','beta':.9}])['metrics']
    produced={(r['method'],r['t']):r for r in produced if r['endpoint']=='checkpoint'}
    for row in original:
        current=produced[(row['method'],row['t'])]
        for field in ['response_mse','response_map_frobenius','posterior_response_excess']:
            assert_allclose(current[field],row[field],rtol=2e-11,atol=2e-13)
        t=row['t'];method=row['method'];posterior=matched[t]
        key={'mean':'M','mode':'Pmode','isotropic':'isotropic','mean_identity':'M'}[method]
        D=(unaligned[t] if method=='mean_identity' else posterior)[key]
        assert_allclose(response_loss(D,data['P'][t-1],query_directions(cfg['v0'])@data['R'][t-1]),row['response_mse'],atol=2e-15)
        assert_allclose(np.linalg.norm(D-data['P'][t-1],'fro')**2,row['response_map_frobenius'],atol=2e-11)


def test_information_anisotropy_fixture_matches_pinned_source():
    actual,checks=anisotropy_fixture(config())
    original=np.load(Path(__file__).parent/'fixtures/audited_contact.npz')
    for name in ['plane_A','plane_mean','normal_logz','iso_normal_A','iso_plane_mean','curvature_eigenvalues','fisher_eigenvalues']:
        assert_allclose(actual[name],original['anisotropy_'+name],rtol=2e-11,atol=2e-11)
    assert checks['numerical_ok']


def test_contact_force_balance_and_real_moment_resolution():
    rng=np.random.default_rng(10431)
    for _ in range(16):
        n=rng.normal(size=3);n/=np.linalg.norm(n);F=rng.normal(size=3)
        velocity,reaction=solve_contact_velocity(F,n,10.)
        assert_allclose(velocity,(np.eye(3)-np.outer(n,n))@F/10,atol=2e-15)
        assert_allclose(10*velocity-n*reaction,F,atol=2e-14)
    for spectrum in [[0.,0.,0.],[0.,2.,6.],[0.,100.,80000.],[0.,80000.,80000.]]:
        A=np.diag(spectrum);result=fast_moments(A,checked=True)
        independent=normal_quadrature(-A,96,512)
        assert result.numerical_ok and result.resolution_checked
        assert result.resolution_error<5e-8
        assert_allclose(result.mean,np.eye(3)-independent['mean'],atol=3e-8)
        assert_allclose(result.log_normalizer,np.trace(A)+independent['log_normalizer'],atol=3e-8)

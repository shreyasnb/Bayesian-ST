"""Execute deterministic real-line geometry and contact soft decisions.

No selected zero-response data are described as a stochastic tracking trial.
The paper profile requires the complete source-current redesign theory gate.
"""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,platform,sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,scipy
from lib.anisotropy_study import (CASE_PARAMETERS,CASE_LABELS,continuous_moments,
    continuous_regions,continuous_region_mass,_continuous_mass,sphere_rule,
    response_summary,soft_action,soft_cost,query_bounds,axial_parameter)
from lib.spectral_series import rank_one_bingham
from lib.reference import rank_one_regions_gr13
from lib.contact_posterior import normal_quadrature

SOURCES=['lib/anisotropy_study.py','scripts/run_anisotropy_study.py','tests/test_anisotropy_study.py',
    'lib/spectral_series.py','lib/reference.py','lib/contact_posterior.py','lib/theory.py','lib/geometry.py']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def portable(value):
    if isinstance(value,dict):return {str(k):portable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [portable(v) for v in value]
    if isinstance(value,np.ndarray):return portable(value.tolist())
    if isinstance(value,np.generic):return portable(value.item())
    return value

def save(path,value):
    path.write_text(json.dumps(portable(value),indent=2,allow_nan=False)+'\n')

def configuration(profile):
    if profile not in ('quick','paper'):raise ValueError('Choose quick or paper')
    return dict(schema=1,profile=profile,N=3,k=1,contact_plane_k=2,
        real_data=True,posterior_mass=.95,soft_penalty=20.,
        A=CASE_PARAMETERS.tolist(),labels=CASE_LABELS,primary_cases=[0,1],
        integration_epsabs=1e-9 if profile=='quick' else 1e-11,
        radial_order=64 if profile=='quick' else 96,
        angular_order=128 if profile=='quick' else 192,
        series_tolerance=1e-11,radii=list(np.linspace(0,1,101)),
        force_angles=list(np.linspace(0,2*np.pi,181)),
        interpretation='Deterministic posterior geometry; quadrature nodes are not independent Monte Carlo repeats.',
        density_measure='q_A with respect to normalized invariant real projective measure; uniform density is one.',
        parameter_semantics='A is the NORMAL-LINE parameter. The contact tangent-plane parameter is -A; psi_plane(-A)=-tr(A)+psi_line(A).',
        primary_control='A1 and A2 have identical mode and entire centered chordal-distance distribution by complementary-coordinate permutation.',
        axial_control='q_(kappa P_mode) matches mean modal overlap and r_rms, not the complete distance CDF or necessarily rho95.',
        region_bound='Bounds are on ||n n^T v||, the normal/blocked component. They are not admissible-plane speed bounds.',
        action='Paper Eq36 for the complementary plane, H=I+lambda E[nn^T]. A soft expected penalty, not a hard contact controller.',
        selection='All matrices, numerical budgets, queries and lambda are fixed by this configuration before evaluation.')


def run(cfg,out):
    out.mkdir(parents=True,exist_ok=True)
    sources={name:sha(ROOT/name) for name in SOURCES}
    key=hashlib.sha256(json.dumps(dict(config=cfg,sources=sources),sort_keys=True).encode()).hexdigest()
    if (out/'manifest.json').exists():
        old=json.loads((out/'manifest.json').read_text())
        if old['source_hash']!=key:raise RuntimeError('Preserve this geometry output; choose a fresh output directory after source changes')
        for name,expected in old['tables'].items():
            if sha(out/name)!=expected:raise RuntimeError('Geometry artifact changed: '+name)
        print('Verified completed geometry cache');return
    gate=None
    if cfg['profile']=='paper':
        from lib.simulation_validation import require_gate
        gate=require_gate(ROOT)
    declaration=dict(config=cfg,sources=sources,source_hash=key,
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        gate=gate,status='running')
    if (out/'preregistered.json').exists():
        if json.loads((out/'preregistered.json').read_text())['source_hash']!=key:
            raise RuntimeError('Frozen geometry sources changed; preserve this directory')
    save(out/'preregistered.json',declaration);start=perf_counter()
    rows=[];directions=[];regions=[];decisions=[];cdf=[]
    means=[];modes=[];logz=[];radii=[];thresholds=[];axial_means=[];kappas=[];nodes=[];weights=[];covariances=[]
    basis=np.eye(3);angles=np.asarray(cfg['force_angles'])
    queries=np.vstack([basis,np.column_stack([np.zeros(len(angles)),np.cos(angles),np.sin(angles)])])
    desired=np.vstack([basis,np.array([0.,1.,1.])/np.sqrt(2),np.ones(3)/np.sqrt(3)])
    for case,A in enumerate(CASE_PARAMETERS):
        native=rank_one_bingham(A,tol=cfg['series_tolerance'])
        continuous=continuous_moments(A,cfg['integration_epsabs'])
        reference=rank_one_regions_gr13(A,mass=cfg['posterior_mass'],epsabs=cfg['integration_epsabs'])
        independent=continuous_regions(A,mass=cfg['posterior_mass'],epsabs=cfg['integration_epsabs'])
        low=sphere_rule(A,cfg['radial_order']//2,cfg['angular_order']//2)
        rule=sphere_rule(A,cfg['radial_order'],cfg['angular_order'])
        contact=normal_quadrature(A,64,192)
        eigenvalues,U=np.linalg.eigh(A);mode=U[:,-1];P=np.outer(mode,mode);M=native.mean;plane=np.eye(3)-M
        kappa,Miso=axial_parameter(M,mode)
        mean_error=float(np.linalg.norm(M-continuous['mean'],2))
        mean_quad_error=float(np.linalg.norm(M-rule['mean'],2))
        logz_error=float(abs(native.log_normalizer-continuous['log_normalizer']))
        reference_error=float(abs(reference.log_normalizer-native.log_normalizer))
        row=dict(case=case,label=CASE_LABELS[case],N=3,k=1,contact_plane_k=2,
            numerical_ok=bool(native.accuracy_met and mean_error<1e-8 and mean_quad_error<1e-8
                and logz_error<1e-8 and reference_error<1e-8
                and abs(independent['radius']-reference.radius)<1e-8
                and abs(independent['hpd_loss']-reference.hpd_loss)<1e-8),
            radius=independent['radius'],hpd_loss=independent['hpd_loss'],
            rms_radius=float(np.sqrt(max(0,1-np.trace(P@M)))),
            axial_kappa=kappa,axial_overlap_error=float(abs(np.trace(P@(M-Miso)))),
            mean_error=mean_error,quadrature_mean_error=mean_quad_error,
            quadrature_resolution_error=float(np.linalg.norm(rule['mean']-low['mean'],2)),
            contact_quadrature_mean_error=float(np.linalg.norm(M-contact['mean'],2)),
            continuous_integration_error=continuous['integration_error'],
            log_normalizer=native.log_normalizer,log_normalizer_error=logz_error,
            dawson_reference_logz_error=reference_error,
            series_order=native.order,series_moment_error_bound=native.moment_error_bound,
            ball_mass=independent['ball_mass'],hpd_mass=independent['hpd_mass'],
            ball_mass_reference=reference.ball_mass,hpd_mass_reference=reference.hpd_mass,
            mass_error=max(abs(independent['ball_mass']-.95),abs(independent['hpd_mass']-.95)),
            mean_0=M[0,0],mean_1=M[1,1],mean_2=M[2,2],
            mean_spectral_anisotropy=float(abs(M[1,1]-M[2,2])))
        rows.append(row);means.append(M);modes.append(mode);logz.append(native.log_normalizer)
        radii.append(independent['radius']);thresholds.append(independent['hpd_loss']);kappas.append(kappa);axial_means.append(Miso)
        nodes.append(rule['nodes']);weights.append(rule['weights'])
        response=response_summary(M,queries)
        covariance_case=[]
        for j,v in enumerate(queries):
            # Direct projector functionals use independent deterministic sphere nodes.
            projected=rule['nodes']*(rule['nodes']@v)[:,None]
            plane_projected=v-projected
            line_error=np.sum((projected-M@v)**2,axis=1)
            plane_error=np.sum((plane_projected-plane@v)**2,axis=1)
            risk=float(response['risk'][j]);quad_risk=float(rule['weights']@line_error)
            iso_excess=float(np.linalg.norm((M-Miso)@v)**2)
            directions.append(dict(case=case,label=CASE_LABELS[case],query=j,
                query_kind='basis' if j<3 else 'tangent-plane circle',
                angle=np.nan if j<3 else angles[j-3],v0=v[0],v1=v[1],v2=v[2],
                mean_normal0=response['mean_normal'][j,0],mean_normal1=response['mean_normal'][j,1],mean_normal2=response['mean_normal'][j,2],
                mean_plane0=response['mean_plane'][j,0],mean_plane1=response['mean_plane'][j,1],mean_plane2=response['mean_plane'][j,2],
                risk=risk,quadrature_risk=quad_risk,plane_quadrature_risk=float(rule['weights']@plane_error),
                modal_excess=float(np.linalg.norm((M-P)@v)**2),axial_excess=iso_excess,
                risk_identity_error=abs(quad_risk-risk),quantity='Total vector risk, not a component variance'))
            if j<3:
                centered=projected-M@v
                covariance_case.append((centered.T*rule['weights'])@centered)
        covariances.append(covariance_case)
        for record in query_bounds(A,independent['radius'],independent['hpd_loss']):
            regions.append(dict(case=case,label=CASE_LABELS[case],mass=.95,
                radius=independent['radius'],hpd_loss=independent['hpd_loss'],**record))
        for desired_index,vdes in enumerate(desired):
            optimal=soft_action(M,vdes,cfg['soft_penalty']);optimal_cost=soft_cost(M,vdes,optimal,cfg['soft_penalty'])
            H=np.eye(3)+cfg['soft_penalty']*M
            candidates={'B':optimal,'mode':soft_action(P,vdes,cfg['soft_penalty']),
                        'axial':soft_action(Miso,vdes,cfg['soft_penalty'])}
            for method,action in candidates.items():
                cost=soft_cost(M,vdes,action,cfg['soft_penalty'])
                identity=float((action-optimal)@H@(action-optimal))
                quad=float(np.sum((action-vdes)**2)+cfg['soft_penalty']*(rule['weights']@((rule['nodes']@action)**2)))
                decisions.append(dict(case=case,label=CASE_LABELS[case],desired=desired_index,method=method,
                    soft_penalty=cfg['soft_penalty'],desired0=vdes[0],desired1=vdes[1],desired2=vdes[2],
                    action0=action[0],action1=action[1],action2=action[2],
                    posterior_cost=cost,quadrature_cost=quad,posterior_excess=cost-optimal_cost,
                    excess_identity=identity,optimality_identity_error=abs(cost-optimal_cost-identity),
                    quadrature_cost_error=abs(cost-quad),
                    scope='Conditional soft decision under the same q_A; not realized controller performance'))
        for radius in cfg['radii']:
            # Use original coordinate order, so the primary swapped pair gets
            # two distinct angular integrals, not identical cached sorted inputs.
            mass=_continuous_mass(float(-A[1,1]),float(-A[2,2]),'ball',radius,cfg['integration_epsabs'])
            cdf.append(dict(case=case,label=CASE_LABELS[case],radius=radius,cdf=mass))
    frames=dict(geometry=pd.DataFrame(rows),directional=pd.DataFrame(directions),
        regions=pd.DataFrame(regions),decisions=pd.DataFrame(decisions),distance_cdf=pd.DataFrame(cdf))
    for name,frame in frames.items():frame.to_csv(out/f'{name}.csv',index=False)
    means,modes,logz,radii,thresholds=map(np.asarray,[means,modes,logz,radii,thresholds])
    arrays=dict(A=CASE_PARAMETERS,mean=means,mean_normal=means,mean_plane=np.eye(3)-means,
        mode=modes,log_normalizer=logz,radius=radii,radius95=radii,hpd_loss=thresholds,
        queries=basis,desired=desired,axial_kappa=np.asarray(kappas),mean_axial_normal=np.asarray(axial_means),
        response_covariance_normal=np.asarray(covariances),
        labels=np.asarray(CASE_LABELS),normal_parameter=True,
        soft_desired=desired[3],soft_penalty=cfg['soft_penalty'],
        soft_action_B=np.asarray([soft_action(M,desired[3],cfg['soft_penalty']) for M in means]),
        soft_action_mode=np.asarray([soft_action(np.outer(h,h),desired[3],cfg['soft_penalty']) for h in modes]),
        soft_action_axial=np.asarray([soft_action(M,desired[3],cfg['soft_penalty']) for M in axial_means]))
    np.savez_compressed(out/'allcases.npz',**arrays)
    primary={name:(value[:2] if isinstance(value,np.ndarray) and value.shape[:1]==(4,) else value)
        for name,value in arrays.items()}
    np.savez_compressed(out/'matrices.npz',**primary)
    np.savez_compressed(out/'posterior_nodes.npz',nodes=np.asarray(nodes),weights=np.asarray(weights),
        interpretation=np.asarray('Deterministic projective quadrature nodes, not Monte Carlo draws'))
    pair=frames['distance_cdf'];a=pair[pair.case==0].cdf.to_numpy();b=pair[pair.case==1].cdf.to_numpy()
    maximum_cdf_error=float(np.max(abs(a-b)))
    checks=dict(numerical_ok=bool(frames['geometry'].numerical_ok.all() and maximum_cdf_error<1e-8
        and frames['directional'].risk_identity_error.max()<1e-8
        and frames['decisions'].optimality_identity_error.max()<1e-8
        and frames['decisions'].quadrature_cost_error.max()<1e-8),
        cases=len(rows),primary_same_distance_cdf_error=maximum_cdf_error,
        primary_radius_difference=float(abs(radii[0]-radii[1])),
        maximum_native_mean_error=float(frames['geometry'].mean_error.max()),
        maximum_native_logz_error=float(frames['geometry'].log_normalizer_error.max()),
        maximum_independent_mass_error=float(frames['geometry'].mass_error.max()),
        maximum_response_risk_identity_error=float(frames['directional'].risk_identity_error.max()),
        maximum_soft_decision_identity_error=float(frames['decisions'].optimality_identity_error.max()),
        maximum_soft_decision_quadrature_error=float(frames['decisions'].quadrature_cost_error.max()),
        maximum_region_dual_gap=float(max(frames['regions'].ball_gap.max(),frames['regions'].hpd_gap.max())),
        seconds=perf_counter()-start,
        scope='Deterministic calculations with independent numerical checks; no frequentist tracking or controller-performance claim')
    save(out/'checks.json',checks)
    if sources!={name:sha(ROOT/name) for name in SOURCES}:raise RuntimeError('Geometry producers changed during execution')
    if not checks['numerical_ok']:
        save(out/'status.json',dict(status='unresolved',checks=checks));raise ArithmeticError('Geometry numerical checks unresolved')
    tables={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['manifest.json','status.json','config.json']}
    document={**declaration,'status':'completed','tables':tables,'checks':checks}
    save(out/'manifest.json',document);save(out/'config.json',document);save(out/'status.json',dict(status='completed',checks=checks))
    print(json.dumps(checks,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['quick','paper'],default='quick');parser.add_argument('--output',type=Path)
    args=parser.parse_args();cfg=configuration(args.profile)
    out=args.output or ROOT/'results/simulation_redesign'/args.profile/'geometry'
    run(cfg,out)

if __name__=='__main__':main()

"""Execute isolated passive tracking and complex matched-process controls."""
from pathlib import Path
import os
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'
import argparse,hashlib,json,platform,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,scipy
from scipy.stats import t as student_t
from lib.beamforming_protocol import output_directory
from lib.beamforming_tracking import (configuration,validation_task,choose_memory,
    long_certificate_task,physical_task,coverage_task,matched_process_data,process_task)
from lib.beamforming_reference import ComplexLineReference,compare_reference
from lib.beamforming_protocol import ula


SOURCES=['lib/beamforming_tracking.py','lib/beamforming_reference.py',
    'lib/beamforming_protocol.py','lib/beamforming_theory.py','lib/beamforming_numerics.py',
    'lib/models.py','lib/geometry.py','lib/theory.py','lib/process_prediction.py',
    'lib/spectral_series.py','benchmarks/case1/baselines.py','scripts/run_beamforming_tracking.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def portable(x):
    if isinstance(x,dict):return {str(k):portable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [portable(v) for v in x]
    if isinstance(x,np.ndarray):return portable(x.tolist())
    if isinstance(x,np.generic):return portable(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x


def digest(x):return hashlib.sha256(json.dumps(portable(x),sort_keys=True,allow_nan=False).encode()).hexdigest()


def save(path,x):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(portable(x),indent=2,allow_nan=False));temporary.replace(path)


def execute(function,tasks,workers,folder,stage,key):
    cache=folder/'checkpoints'/stage;cache.mkdir(parents=True,exist_ok=True)
    done={};pending=[]
    for i,task in enumerate(tasks):
        path=cache/f'{i:05d}.json';task_hash=digest(task)
        try:
            record=json.loads(path.read_text())
            assert record['source_hash']==key and record['task_hash']==task_hash
            assert record['result_hash']==digest(record['result']);done[i]=record['result']
        except (OSError,ValueError,KeyError,AssertionError):pending.append((i,task,task_hash,path))
    print(stage,'restored',len(done),'/',len(tasks),flush=True)
    def retain(i,result,task_hash,path):
        arrays=result.pop('arrays',None) if isinstance(result,dict) else None
        if arrays:
            target=folder/'trajectories';target.mkdir(exist_ok=True)
            np.savez_compressed(target/f'{stage}_{i:05d}.npz',**arrays)
        result=portable(result);save(path,dict(source_hash=key,task_hash=task_hash,result_hash=digest(result),result=result))
        done[i]=result
        if len(done)%8==0 or len(done)==len(tasks):print(stage,'completed',len(done),'/',len(tasks),flush=True)
    if workers==1:
        for i,task,task_hash,path in pending:retain(i,function(task),task_hash,path)
    elif pending:
        with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
            futures={pool.submit(function,task):(i,task_hash,path) for i,task,task_hash,path in pending}
            for future in as_completed(futures):
                i,task_hash,path=futures[future];retain(i,future.result(),task_hash,path)
    return [done[i] for i in range(len(tasks))]


def source_hashes():return {name:sha(ROOT/name) for name in SOURCES}


def register(cfg,out,component):
    sources=source_hashes();versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
    key=digest(dict(config=cfg,sources=sources,versions=versions,component=component))
    document=dict(config=cfg,sources=sources,versions=versions,source_hash=key,component=component,
        hardware=dict(platform=platform.platform(),cpu_count=os.cpu_count(),blas_threads=1,
            cgroup_cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip() if Path('/sys/fs/cgroup/cpu.max').exists() else None),
        numerical_status='Mean, state and scalar-tail flags are separate; missing tails are not resolved tails.')
    out.mkdir(parents=True,exist_ok=True)
    path=out/'preregistered.json'
    if path.exists() and json.loads(path.read_text())['source_hash']!=key:
        raise RuntimeError('Preserve existing protocol; scientific sources/configuration differ')
    save(path,document);return key,document


def completed(out,key):
    path=out/'manifest.json'
    if not path.exists():return False
    old=json.loads(path.read_text())
    if old['source_hash']!=key:raise RuntimeError('Completed evidence differs; choose a new study directory')
    for name,digest_value in old['tables'].items():
        if sha(out/name)!=digest_value:raise RuntimeError('Completed table hash changed: '+name)
    print('Verified completed',out,flush=True);return True


def freeze_physical(cfg,out,key,workers):
    tasks=[(cfg,s,j) for s in cfg['noises'] for j in range(cfg['validation_trials'])]
    results=execute(validation_task,tasks,workers,out,'validation',key)
    rows=[dict(r,stage='initial') for group in results for r in group]
    selected,_=choose_memory(rows,cfg);expansion=[]
    if any(r['beta']==min(cfg['betas']) for r in selected):
        expansion=[.8,.6,.4];extra=dict(cfg,betas=expansion)
        more=execute(validation_task,[(extra,s,j) for s in cfg['noises'] for j in range(cfg['validation_trials'])],
            workers,out,'validation_expansion',key)
        rows.extend(dict(r,stage='single boundary extension') for group in more for r in group)
    all_cfg=dict(cfg,betas=cfg['betas']+expansion);selected,curves=choose_memory(rows,all_cfg)
    pd.DataFrame(rows).to_csv(out/'validation_trials.csv',index=False);curves.to_csv(out/'tuning.csv',index=False)
    document=dict(selected=selected,source_hash=key,expansion=expansion,
        selection_completed_before_test=True,test_observations_used=False,
        boundary_status='One declared extension only; any remaining boundary is reported, not extended on test.')
    save(out/'selected.json',document);return document


def require_paper_gate(cfg):
    if cfg['name']=='paper':
        from lib.beamforming_validation import require_beamforming_gate
        return require_beamforming_gate(ROOT)
    return dict(status='engineering quick profile; not the paper comparison')


def finalize(out,document,frames,gate,start,extra=None):
    if source_hashes()!=document['sources']:raise RuntimeError('Scientific sources changed during run')
    for name,frame in frames.items():frame.to_csv(out/f'{name}.csv',index=False)
    names=[p for p in out.rglob('*') if p.is_file() and 'checkpoints' not in p.parts
        and p.name not in ['manifest.json','status.json','findings.md']]
    tables={str(p.relative_to(out)):sha(p) for p in sorted(names)}
    checks=dict(seconds=perf_counter()-start,rows={k:len(v) for k,v in frames.items()})
    if extra:checks.update(extra)
    save(out/'manifest.json',dict(**document,tables=tables,gate=gate,status='completed',checks=checks))
    save(out/'status.json',dict(status='completed',source_hash=document['source_hash'],checks=checks))


def tracking_study(cfg,out,workers,validation_only=False,test_only=False):
    key,document=register(cfg,out,'tracking')
    if completed(out,key):return
    if test_only:
        selection=json.loads((out/'selected.json').read_text())
        if selection['source_hash']!=key:raise RuntimeError('Frozen beta selection differs from current source')
    else:selection=freeze_physical(cfg,out,key,workers)
    if validation_only:return
    gate=require_paper_gate(cfg);save(out/'test_started.json',dict(gate=gate,selected_hash=sha(out/'selected.json'),source_hash=key))
    save(out/'status.json',dict(status='running',stage='heldout',source_hash=key));start=perf_counter()
    long=execute(long_certificate_task,[(cfg,j) for j in range(cfg['long_trials'])],workers,out,'long',key)
    physical=execute(physical_task,[(cfg,s,j,selection['selected']) for s in cfg['noises'] for j in range(cfg['test_trials'])],workers,out,'physical',key)
    coverage=execute(coverage_task,[(cfg,state,j) for state in cfg['coverage_states'] for j in range(cfg['coverage_trials'])],workers,out,'coverage',key)
    frames={name:pd.DataFrame([r for group in physical for r in group[name]]) for name in ['metrics','tails','scan','regional']}
    frames['bounds']=pd.DataFrame([r for group in long for r in group]+[r for group in physical for r in group['bounds']])
    frames['coverage']=pd.DataFrame([r for group in coverage for r in group])
    violations=int(frames['bounds'].certificate_violated.sum())
    mean_failures=int((~frames['metrics'].mean_ok.astype(bool)).sum())
    coverage_failures=int((~frames['coverage'].state_ok.astype(bool)).sum())
    coverage_scalar_failures=int((~frames['coverage'].tail_ok.astype(bool)).sum())
    tail_failures=int((~frames['tails'].numerical_ok.astype(bool)).sum())
    finalize(out,document,frames,gate,start,dict(valid_bounds=int(frames['bounds'].valid.sum()),bound_violations=violations,
        mean_failures=mean_failures,coverage_state_failures=coverage_failures,
        coverage_functional_tail_failures=coverage_scalar_failures,physical_tail_failures=tail_failures,
        maximum_regional_dual_gap=float(frames['regional'].dual_gap.max())))
    means=frames['metrics'].groupby(['s','t','policy','method']).population_power.mean().reset_index()
    means.to_csv(out/'summary.csv',index=False)
    bounds=frames['bounds'].groupby(['condition','s','t','beta'])[['valid','certificate_violated','actual_error','eta','drift','bound']].mean()
    text=['# Passive modal tracking','',
        'A deterministic ULA arrival path and passive proper-complex observations are used. Identity transport is practical; no true rotations or latent coefficients enter estimation. Known powers are retained. The long stationary cohort is distinct from the short physical experiment.','',
        f'Valid fixed-time certificates: {int(frames["bounds"].valid.sum())}; violations: {violations}. Invalid sufficient conditions remain NaN rather than clipped successes. Bounds use deterministic scoring drift and a quadratic Gaussian covariance-noise bound, not a posterior radius.','',
        f'Mean failures: {mean_failures}. Stationary native-array state-region reports unresolved: {coverage_failures}; their separate scalar-functional reports unresolved: {coverage_scalar_failures}. Physical-subset uncertainty reports unresolved: {tail_failures}. Tail Monte Carlo precision is not implied by a resolved mean.','',
        'The beta family is selected independently per known noise level on terminal modal beamformer power. A single predeclared lower-bound extension is recorded in selected.json. The same selected beta is used by the posterior mean, modal and matched-isotropization decisions. The all-history comparison uses beta=1.','',
        'Reported regions on a stationary fixed-state cohort test the passive unitary-invariance specialization. Physical ULA motion plus forgetting is outside that coverage argument. Scalar power bands, common HPD-region envelopes, and projection-risk diagnostics answer different questions.','',
        'trajectory records are preassigned trial zero, one for each noise regime. Scanning response s+q*a^H P a is distinct from the weight beampattern |w^H a|^2.','',
        bounds.to_string(),'']
    (out/'findings.md').write_text('\n'.join(text))
    # Add the compact summary and findings after writing them, without omitting their hashes.
    manifest=json.loads((out/'manifest.json').read_text())
    for name in ['summary.csv','findings.md']:manifest['tables'][name]=sha(out/name)
    save(out/'manifest.json',manifest)


def reference_development(cfg,out):
    folder=out/'reference_development';folder.mkdir(parents=True,exist_ok=True);rows=[]
    queries=np.asarray([ula(2,np.deg2rad(a)) for a in [-45,-20,0,20,45]])
    for kappa in cfg['process_kappas']:
        for trial in range(cfg['reference_development_trials']):
            data=matched_process_data(cfg,cfg['process_development_seed'],trial,2,kappa)
            low=ComplexLineReference(kappa,**cfg['reference_coarse']);high=ComplexLineReference(kappa,**cfg['reference_fine'])
            for t,u in enumerate(data['U'],1):
                low.update(u,cfg['process_s'],cfg['q']);high.update(u,cfg['process_s'],cfg['q'])
                if t in cfg['process_reports']:
                    _,audit=compare_reference(low,high,queries,data['h'][t-1])
                    rows.append(dict(kappa=kappa,trial=trial,t=t,**audit))
            print('CP1 reference development',kappa,trial,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(folder/'checks.csv',index=False)
    status=dict(status='passed' if frame.reference_numerical_ok.all() else 'unresolved',
        targets=len(frame),failures=int((~frame.reference_numerical_ok.astype(bool)).sum()),
        maxima={name:float(frame[name].max()) for name in ['density_tv','mean_error','scalar_cdf_error','quantile_width_relative_error','region_mass_error']},
        budgets=dict(coarse=cfg['reference_coarse'],fine=cfg['reference_fine']),seed=cfg['process_development_seed'],
        source_hashes={name:sha(ROOT/name) for name in ['lib/beamforming_reference.py','scripts/run_beamforming_tracking.py']},
        test_data_used=False,numerical_scope='Empirical harmonic/grid refinement, not rigorous all-density enclosures')
    save(folder/'status.json',status);return status


def process_study(cfg,out,workers,development_only=False):
    key,document=register(cfg,out,'process')
    if completed(out,key):return
    development=reference_development(cfg,out)
    if development_only:return
    if development['status']!='passed':raise ArithmeticError('Reference development unresolved; preserve it before revising a numerical budget')
    gate=require_paper_gate(cfg);save(out/'test_started.json',dict(gate=gate,development=development,source_hash=key))
    save(out/'status.json',dict(status='running',stage='heldout',source_hash=key));start=perf_counter()
    tasks=[(cfg,n,kappa,j) for n in cfg['process_n'] for kappa in cfg['process_kappas'] for j in range(cfg['process_trials'])]
    results=execute(process_task,tasks,workers,out,'matched',key)
    frames={name:pd.DataFrame([r for group in results for r in group[name]]) for name in ['metrics','reference_checks','shape_checks','scan','scores']}
    m=frames['metrics'];tail=m[m.tail_evaluated.fillna(False).astype(bool)]
    finalize(out,document,frames,gate,start,dict(mean_failures=int((~m.mean_ok.astype(bool)).sum()),
        native_tail_targets=int((tail.method!='Full-density CP1 reference').sum()),
        native_tail_failures=int((~tail[tail.method!='Full-density CP1 reference'].numerical_ok.astype(bool)).sum()),
        reference_targets=len(frames['reference_checks']),reference_failures=int((~frames['reference_checks'].reference_numerical_ok.astype(bool)).sum()),
        maximum_inverse_residual=float(m.inverse_residual.max())))
    summary=m[m.t==cfg['process_T']].groupby(['n','kappa','method']).agg(
        trials=('trial','nunique'),tail_trials=('tail_evaluated','sum'),population_power=('population_power','mean'),projector_mean_mse=('projector_mean_mse','mean'),
        clean_coverage=('clean_coverage','mean'),clean_width=('clean_width','mean'),interval_score=('interval_score','mean')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    findings=['# Matched complex-line process','',
        'The native n=8 process and low-dimensional CP1 reference are separate cohorts under the same passive observation and distortionless decision model. They are not deterministic ULA motion. Kappa is known in this comparison.','',
        'Process Bingham matches the exact predictive first moment, then projects the predictive density onto one Bingham family. Identity and beta=0.95 power forgetting are fixed diagnostic controls, not optimal tuned competitors.','',
        f'Full CP1 reference subset: first {cfg["reference_trials"]} trajectories per kappa, fixed before test outcomes. All harmonic degrees are used, with a complex Bloch-sphere von Mises--Fisher kernel. It is not a real Watson calculation or an n=8 reference.','',
        'Continuous Legendre integration gives quadratic-functional CDFs and mean-eigenspace ball radii for the reference. Density HPD thresholds use the declared sphere grid, with independent refinement flags. Failed targets remain in the tables.','',
        summary.to_string(index=False),'']
    (out/'findings.md').write_text('\n'.join(findings))
    manifest=json.loads((out/'manifest.json').read_text())
    for name in ['summary.csv','findings.md']:manifest['tables'][name]=sha(out/name)
    save(out/'manifest.json',manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['quick','paper'],default='quick')
    parser.add_argument('--component',choices=['all','tracking','process'],default='all')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--validation-only',action='store_true');parser.add_argument('--test-only',action='store_true')
    parser.add_argument('--reference-development-only',action='store_true')
    parser.add_argument('--output-root',type=Path)
    args=parser.parse_args();cfg=configuration(args.profile)
    base=args.output_root or ROOT/'results/beamforming_unified'/args.profile
    if args.validation_only and args.test_only:parser.error('Choose only one validation/test stage')
    if args.component in ['all','tracking'] and not args.reference_development_only:
        tracking_study(cfg,base/'tracking',args.workers,args.validation_only,args.test_only)
    if args.component in ['all','process'] and not args.validation_only:
        process_study(cfg,base/'process',args.workers,args.reference_development_only)


if __name__=='__main__':main()

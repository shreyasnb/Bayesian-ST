"""Execute the draft-5 passive-array geometry study with frozen validation."""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import gzip,hashlib,json,platform,sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import scipy
from lib.array_geometry_study import configuration,validation_task,select_policies,test_task,paired_table,fixture

SOURCES=['lib/array_geometry_study.py','scripts/run_array_geometry.py','tests/test_array_geometry.py',
    'lib/beamforming_numerics.py','lib/beamforming_decisions.py','lib/beamforming_protocol.py',
    'lib/geometry.py','lib/models.py','lib/moments.py','lib/spectral_series.py','lib/process_prediction.py',
    'benchmarks/case1/baselines.py','benchmarks/case2/baselines.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def portable(x):
    if isinstance(x,dict):return {str(k):portable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [portable(v) for v in x]
    if isinstance(x,np.ndarray):return portable(x.tolist())
    if isinstance(x,np.generic):return portable(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def digest(x):return hashlib.sha256(json.dumps(portable(x),sort_keys=True,allow_nan=False).encode()).hexdigest()
def save(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(portable(value),indent=2,allow_nan=False)+'\n');temporary.replace(path)


def execute(function,tasks,out,stage,source_hash,workers):
    folder=out/'checkpoints'/stage;folder.mkdir(parents=True,exist_ok=True)
    complete={};pending=[]
    for i,task in enumerate(tasks):
        path=folder/f'{i:05d}.json';signature=digest(task)
        try:
            old=json.loads(path.read_text())
            assert old['source_hash']==source_hash and old['task_hash']==signature
            assert old['result_hash']==digest(old['result'])
            for name,expected in old.get('artifacts',{}).items():assert sha(out/name)==expected
            complete[i]=old['result']
        except (OSError,ValueError,KeyError,AssertionError):pending.append((i,task,signature,path))
    print(stage,'restored',len(complete),'/',len(tasks),flush=True)
    def retain(i,result,signature,path):
        arrays=result.pop('arrays',{}) if isinstance(result,dict) else {};artifacts={}
        if arrays:
            target=out/'stories'/f'{stage}_{i:05d}.npz';target.parent.mkdir(exist_ok=True)
            np.savez_compressed(target,**arrays);artifacts[str(target.relative_to(out))]=sha(target)
        result=portable(result)
        save(path,dict(source_hash=source_hash,task_hash=signature,result_hash=digest(result),artifacts=artifacts,result=result))
        complete[i]=result
        if len(complete)%16==0 or len(complete)==len(tasks):print(stage,'completed',len(complete),'/',len(tasks),flush=True)
    if workers==1:
        for i,task,signature,path in pending:retain(i,function(task),signature,path)
    elif pending:
        with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
            futures={pool.submit(function,task):(i,signature,path) for i,task,signature,path in pending}
            for future in as_completed(futures):
                i,signature,path=futures[future];retain(i,future.result(),signature,path)
    return [complete[i] for i in range(len(tasks))]


def summarize(frames):
    m=frames['metrics'];tail=frames['tails'];shape=frames['shape']
    check=dict(metric_rows=len(m),trajectories=int(m[['n','law','s','trial']].drop_duplicates().shape[0]),
        mean_failures=int((~m.mean_ok.astype(bool)).sum()),tail_reports=len(tail),
        unresolved_tail_reports=int((~tail.numerical_ok.astype(bool)).sum()),
        unresolved_state_reports=int((~tail.state_ok.astype(bool)).sum()),
        maximum_unit_gain_error=float(m.unit_gain_error.max()),
        maximum_expected_power_identity_error=float(m.optimality_residual.max()),
        one_snapshot_maximum_anisotropy=float(shape.loc[shape.t.eq(1),'anisotropy'].max()),maximum_independent_pca_error=float(shape.pca_mode_error.max()),maximum_parameter_scatter_error=float(shape.parameter_scatter_error.max()))
    summary=m.groupby(['n','law','s','t','method']).agg(trials=('trial','nunique'),
        true_power=('true_power','mean'),heldout_power=('heldout_power','mean'),response_loss=('response_loss','mean'),
        posterior_risk=('posterior_risk','mean'),posterior_excess=('posterior_excess','mean'),
        mean_seconds=('mean_seconds','mean')).reset_index()
    return check,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['quick','paper'],default='quick')
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--output',type=Path)
    parser.add_argument('--validation-only',action='store_true');parser.add_argument('--test-only',action='store_true')
    args=parser.parse_args()
    if args.validation_only and args.test_only:parser.error('Choose one stage only')
    cfg=configuration(args.profile)
    out=(args.output or ROOT/'results/simulation_redesign'/args.profile/'beamforming').resolve();out.mkdir(parents=True,exist_ok=True)
    gate=None
    if args.profile=='paper':
        from lib.simulation_validation import require_gate
        gate=require_gate(ROOT)
    sources={name:sha(ROOT/name) for name in SOURCES};versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
    key=digest(dict(config=cfg,sources=sources,versions=versions))
    if (out/'manifest.json').exists():
        old=json.loads((out/'manifest.json').read_text())
        if old['source_hash']==key and all(sha(out/name)==expected for name,expected in old['tables'].items()):
            print('Verified completed array-geometry evidence');return
        raise RuntimeError('Preserve completed evidence and choose a fresh output directory')
    hardware=dict(platform=platform.platform(),logical_cpus=os.cpu_count(),blas_threads=1,
        cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip() if Path('/sys/fs/cgroup/cpu.max').exists() else None)
    declaration=dict(config=cfg,sources=sources,versions=versions,source_hash=key,profile=args.profile,
        gate=gate,hardware=hardware,workers=args.workers)
    if (out/'preregistered.json').exists() and json.loads((out/'preregistered.json').read_text())['source_hash']!=key:
        raise RuntimeError('Sources changed during this run; preserve it before restarting elsewhere')
    save(out/'preregistered.json',declaration)
    start=perf_counter()
    if not args.test_only:
        save(out/'status.json',dict(status='running',stage='validation',source_hash=key))
        tasks=[(cfg,c['n'],law,s,j) for c in cfg['cohorts'] for law in cfg['laws'] for s in cfg['noises'] for j in range(c['validation_trials'])]
        grouped=execute(validation_task,tasks,out,'validation',key,args.workers);records=[row for group in grouped for row in group]
        selected,curves=select_policies(records,cfg)
        # Keep all cached spectral CV statistics, not only the chosen settings.
        with gzip.open(out/'validation_statistics.json.gz','wt') as stream:json.dump(portable(records),stream,allow_nan=False)
        pd.DataFrame(curves).to_csv(out/'validation_curves.csv',index=False)
        save(out/'selected.json',dict(source_hash=key,selected=selected,
            validation_completed_before_test=True,test_data_used=False,
            objective='Mean true-covariance output power on independent validation data',
            qualification='Loading/scalar use continuous one-dimensional search; GROUSE is the fixed nine-step grid. Boundaries remain reported.'))
    else:
        frozen=json.loads((out/'selected.json').read_text())
        if frozen['source_hash']!=key:raise RuntimeError('Frozen selection source differs')
        selected=frozen['selected']
    if args.validation_only:return
    selection_sha=sha(out/'selected.json')
    save(out/'test_preregistered.json',dict(**declaration,selection_sha256=selection_sha,heldout_opened_after_selection=True))
    save(out/'status.json',dict(status='running',stage='heldout',source_hash=key))
    tasks=[(cfg,c['n'],law,s,j,selected) for c in cfg['cohorts'] for law in cfg['laws'] for s in cfg['noises'] for j in range(c['test_trials'])]
    grouped=execute(test_task,tasks,out,'test',key,args.workers)
    frames={name:pd.DataFrame([row for group in grouped for row in group[name]]) for name in ['metrics','shape','tails','scans','spectra']}
    frames['paired']=paired_table(frames['metrics']);checks,frames['summary']=summarize(frames)
    for name,frame in frames.items():frame.to_csv(out/f'{name}.csv',index=False)
    arrays={}
    for path in sorted((out/'stories').glob('test_*.npz')):
        with np.load(path,allow_pickle=False) as data:
            for name in data.files:
                if name in arrays and not np.array_equal(arrays[name],data[name]):raise RuntimeError('Conflicting preassigned story arrays')
                arrays[name]=data[name]
    np.savez_compressed(out/'stories.npz',**arrays)
    fixture_rows,fixture_arrays,fixture_checks=fixture(cfg)
    pd.DataFrame(fixture_rows).to_csv(out/'fixture.csv',index=False)
    np.savez_compressed(out/'fixture.npz',**fixture_arrays);save(out/'fixture_checks.json',fixture_checks)
    if any(sha(ROOT/name)!=expected for name,expected in sources.items()):raise RuntimeError('Scientific sources changed during generation')
    checks.update(total_seconds=perf_counter()-start,fixture=fixture_checks,
        uncertainty_scope='Mean/normalizer accuracy, iid Monte Carlo reports, and repeated-data coverage are separate checks; all flags retained.')
    write_findings(out,cfg,checks,frames)
    files=[p for p in out.rglob('*') if p.is_file() and 'checkpoints' not in p.parts and p.name not in ['manifest.json','status.json']]
    tables={str(p.relative_to(out)):sha(p) for p in sorted(files)}
    save(out/'manifest.json',dict(**declaration,status='completed',tables=tables,checks=checks))
    save(out/'status.json',dict(status='completed',source_hash=key,checks=checks))
    print(json.dumps(portable(checks),indent=2,allow_nan=False),flush=True)


def write_findings(out,cfg,checks,frames):
    lines=['# Passive complex-array geometry','',
        f"Profile: {cfg['profile']}. Primary n=8 has {cfg['cohorts'][0]['validation_trials']} independent validation and {cfg['cohorts'][0]['test_trials']} test trajectories per law/noise. The n=16 cohort has its own recorded denominator. No source from the draft simulations section defines this protocol.",'',
        'The stationary ULA desired arrival is0deg and interferer separations are3,10,30deg; the Haar signature is the matched-prior control. Uniform ambient complex-line inference is not a posterior restricted to physical arrival angle. Known s,tau2 and beta1 are shared across point/mean decisions.','',
        'SCM diagonal loading is selected continuously via alpha in[0,1]; alpha1 is its exact isotropic endpoint. The separate Ledoit-Wolf reference adapts its shrinkage to each observed sample covariance. Its one-snapshot convention is explicitly alpha1. Global modal covariance regularization and per-posterior modal-overlap matching are different controls.','',
        'Methods receive identical passive snapshots; the desired steering is used only for decisions. Validation losses use known simulated covariances on independent episodes, never test truth for tuning. All methods share held-out measurements within a test trajectory.','',
        f"Mean failures: {checks['mean_failures']}; unresolved native reports: {checks['unresolved_tail_reports']}/{checks['tail_reports']}. Finite iid Monte Carlo precision is not a deterministic tail certificate. Scalar intervals at an exactly aligned ULA query can have zero fixed-state coverage at the support boundary; this does not contradict stationary state-region invariance.",'',
        'The selected-data fixture is a structural calculation. Its two Bingham parameters have the same spectrum and mode, related by a unitary fixing the mode; hence every centered chordal-distance quantile agrees. Their posterior mean matrices and optimal expected-power weights differ. Synthetic two-snapshot blocks recover the parameters exactly up to the identity gauge. These are not typical random-training outcomes.','',
        'All true-power, held-out-power and derived response comparisons remain in paired.csv, including losses. The independent trajectory is the repetition unit. Response queries and scan-grid points are not extra repetitions. Methods supplying only a covariance do not acquire an invented projector-response prediction: their response_loss is undefined.','',
        'The scan bands use preassigned trial0, ULA10deg, s=.2. Densely displayed pointwise quantiles are Monte Carlo summaries from native iid draws; only the five fixed query directions carry the declared quantile-MCSE checks. Saved plot clouds use a predeclared subset of those draws.','',
        '## References','',
        'Ledoit and Wolf (2004), A well-conditioned estimator for large-dimensional covariance matrices, Journal of Multivariate Analysis93:365–411. The implementation uses the uncentered outer-product Frobenius form with Hermitian products for proper complex observations.','',
        'Balzano, Nowak and Recht (2010), Online identification and tracking of subspaces from highly incomplete information. GROUSE here uses complete observations and the same known-power decision covariance.','']
    (out/'findings.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()

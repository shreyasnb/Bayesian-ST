"""Execute the draft-5 contact-normal study with frozen independent cohorts."""
from pathlib import Path
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from time import perf_counter
import argparse,hashlib,json,platform,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import scipy
from lib.contact_geometry_study import (configuration,static_validation,select_static,static_test,
    dynamic_validation,select_dynamic,dynamic_test,higher_test,certificate_test,dynamic_certificates,
    build_stories)

SOURCES=['lib/contact_geometry_study.py','scripts/run_contact_geometry.py','tests/test_contact_geometry.py',
    'lib/models.py','lib/geometry.py','lib/spectral_series.py','lib/contact_mechanics.py',
    'lib/contact_posterior.py','lib/contact_process_prediction.py','lib/contact_gaussian.py',
    'lib/reference.py','lib/theory.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def portable(value):
    if isinstance(value,dict):return {str(k):portable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [portable(x) for x in value]
    if isinstance(value,np.ndarray):return portable(value.tolist())
    if isinstance(value,np.generic):return portable(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value

def digest(value):return hashlib.sha256(json.dumps(portable(value),sort_keys=True,allow_nan=False).encode()).hexdigest()
def save(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(portable(value),indent=2,allow_nan=False)+'\n');temporary.replace(path)


def run_jobs(function,tasks,out,stage,key,workers):
    folder=out/'checkpoints'/stage;folder.mkdir(parents=True,exist_ok=True);done={};pending=[]
    for index,task in enumerate(tasks):
        path=folder/f'{index:05d}.json';signature=digest(task)
        try:
            previous=json.loads(path.read_text())
            assert previous['source_hash']==key and previous['task_hash']==signature
            assert digest(previous['result'])==previous['result_hash'];done[index]=previous['result']
        except (OSError,ValueError,KeyError,AssertionError):pending.append((index,task,signature,path))
    print(stage,'restored',len(done),'/',len(tasks),flush=True)
    def retain(index,value,signature,path):
        value=portable(value);save(path,dict(source_hash=key,task_hash=signature,result_hash=digest(value),result=value));done[index]=value
        if len(done)%8==0 or len(done)==len(tasks):print(stage,'completed',len(done),'/',len(tasks),flush=True)
    if workers==1:
        for index,task,signature,path in pending:retain(index,function(task),signature,path)
    elif pending:
        with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
            futures={pool.submit(function,task):(index,signature,path) for index,task,signature,path in pending}
            for future in as_completed(futures):
                index,signature,path=futures[future];retain(index,future.result(),signature,path)
    return [done[i] for i in range(len(tasks))]


def dynamic_job(task):
    result=dynamic_test(task);result.pop('story',None)
    cfg,condition,trial,selection=task
    result['bounds']=dynamic_certificates(cfg,condition,trial,selection)
    return result


def write_findings(out,cfg,frames,checks):
    static=frames['paired_static'];dynamic=frames['paired_dynamic']
    lines=['# Contact-normal study','',
        'The observations use the real conditional model u=P_X v+e. P_X is a tangent-plane projector; its normal line has Bingham parameter -A. Known applied free velocity and measured wrist rotation are supplied to every corresponding estimator. True surface normals and positions are scoring-only.','',
        f"The {cfg['profile']} profile has {cfg['static_validation_trials']} validation and {cfg['static_test_trials']} independent test datasets per static input program, {cfg['dynamic_validation_trials']} validation and {cfg['dynamic_test_trials']} test trajectories per physical condition. Query coordinates and reporting times are not independent experimental repeats.",'',
        'Every static command has the same Euclidean norm. Balanced and directionally concentrated programs therefore have identical total energy at every horizon. Their local Fisher information need not be the same. The stationary normal is Haar-distributed in the Monte Carlo comparison; the fixed e3 density illustration is a separate preassigned example.','',
        '## Comparators and decisions','',
        'The modal plane equals independent weighted conditional projector least squares. The mean, mode and matched isotropized normal mean use identical A in the shape comparison. A separate globally validated scalar contracts the modal plane toward the known-rank isotropic mean. Ridge regression also regularizes toward that known-rank isotropic mean, rather than zero; it uses force and measured velocity. Residual PCA uses v-u and therefore has access to both signals, while discarding some conditional-likelihood information.','',
        'The same-parameter local Gaussian uses the modal Hessian of -A. Its exponential-map mean is integrated; local cut-locus mass and numerical refinement have separate flags. Tied precision fails explicitly. The physical recursive Gaussian uses a six-observation data-only initialization, Joseph correction and the same measured frame rotations; Q is selected separately on validation. It is not the same-parameter control.','',
        'For the physical contact, each method selects its own beta or Q on independent terminal response loss. The mean and matched-isotropization rows share beta by construction. The identity-frame ablation retains the same beta and data. G=R_t^T R_(t-1) is derived from wrist orientation independent of the hidden surface normal, not an oracle normal alignment.','',
        'The expected decision is v^B=[I+lambda(I-Pbar)]^(-1) v_des. Cost uses a soft inadmissible-command penalty, not a guarantee of feasible contact motion or a closed-loop controller. The expected penalty is linear in I-Pbar. Unconstrained ridge maps are scored for response prediction only and have no Eq.36 cost.','',
        '## Static paired response comparisons','',static.to_string(index=False),'',
        '## Physical paired response comparisons','',dynamic.to_string(index=False),'',
        '## Numerical and interpretation limits','',json.dumps(portable(checks),indent=2),'',
        'Posterior conditional mass, numerical integration accuracy and repeated-data inclusion are different claims. Forgetting under a changing normal does not inherit a stationary confidence guarantee. Scalar intervals are clean-response quantiles, not noisy velocity prediction intervals.','',
        'The long stationary cohort checks the conditional modal theorem through residual r=v-u=(nn^T)v-e. Its normal-line mode is the complementary tangent-plane mode, and chordal errors agree. Bounds use known noise and deterministic inputs/transport and are pointwise at predeclared times. Truth-dependent drift is an evaluation quantity, not an online observable certificate. Invalid conditions remain NaN.','',
        'Higher-dimensional examples are generalized maintained single linear constraints under isotropic resistance. They do not silently claim a seven-joint robot whose known Jacobian would be additional information. Normal chordal error and dimension-normalized response error are separate, avoiding artificial improvement simply because a single constraint occupies a smaller fraction of a large space.','',
        'All original contact results remain unchanged. Full worker checkpoints are regenerable from seeds; compact metrics, selections, story arrays and every numerical/local-validity flag are saved. Wall clocks during parallel generation are contended and do not establish hardware-independent complexity.']
    (out/'findings.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['quick','paper'],default='quick')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();cfg=configuration(args.profile)
    out=(args.output or ROOT/'results/simulation_redesign'/args.profile/'contact').resolve();out.mkdir(parents=True,exist_ok=True)
    sources={name:sha(ROOT/name) for name in SOURCES};versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
    key=digest(dict(config=cfg,sources=sources,versions=versions))
    if (out/'manifest.json').exists():
        previous=json.loads((out/'manifest.json').read_text())
        if previous['source_hash']==key and all(sha(out/name)==value for name,value in previous['tables'].items()):
            print('Verified completed contact evidence');return
        raise RuntimeError('Preserve completed evidence; use a fresh output directory after a source change')
    if args.profile=='paper':
        from lib.simulation_validation import require_gate
        gate=require_gate(ROOT)
    else:gate=dict(status='Engineering quick profile; no paper comparison claim')
    declaration=dict(config=cfg,sources=sources,versions=versions,source_hash=key,gate=gate,
        workers=args.workers,hardware=dict(platform=platform.platform(),logical_cpus=os.cpu_count(),blas_threads=1),
        base_commit='2a8697c7e7fa27b3779f3abeaed32ca8b323a9ef',status='running')
    if (out/'preregistered.json').exists() and json.loads((out/'preregistered.json').read_text())['source_hash']!=key:
        raise RuntimeError('Frozen contact sources changed; preserve the partial run and choose another output')
    save(out/'preregistered.json',declaration);save(out/'status.json',dict(status='running',stage='validation',source_hash=key))
    start=perf_counter()
    task=[(cfg,kind,j) for kind in cfg['static_inputs'] for j in range(cfg['static_validation_trials'])]
    validation=[row for group in run_jobs(static_validation,task,out,'static_validation',key,args.workers) for row in group]
    static_selection=select_static(validation,cfg)
    dynamic_task=[(cfg,condition,j) for condition in cfg['dynamic_conditions'] for j in range(cfg['dynamic_validation_trials'])]
    dynamic_validation_rows=[row for group in run_jobs(dynamic_validation,dynamic_task,out,'dynamic_validation',key,args.workers) for row in group]
    dynamic_selection=select_dynamic(dynamic_validation_rows,cfg)
    save(out/'selected.json',dict(source_hash=key,static=static_selection,dynamic=dynamic_selection,
        selection_completed_before_test=True,test_data_used=False,criterion='Terminal complete-vector response loss on independent validation trajectories'))
    pd.DataFrame(validation).to_csv(out/'static_validation.csv',index=False)
    pd.DataFrame(dynamic_validation_rows).to_csv(out/'dynamic_validation.csv',index=False)
    save(out/'test_preregistered.json',dict(**declaration,selected_sha256=sha(out/'selected.json'),heldout_opened_after_selection=True))
    save(out/'status.json',dict(status='running',stage='test',source_hash=key))
    static_task=[(cfg,kind,j,static_selection) for kind in cfg['static_inputs'] for j in range(cfg['static_test_trials'])]
    static_groups=run_jobs(static_test,static_task,out,'static_test',key,args.workers)
    dynamic_task=[(cfg,condition,j,dynamic_selection) for condition in cfg['dynamic_conditions'] for j in range(cfg['dynamic_test_trials'])]
    dynamic_groups=run_jobs(dynamic_job,dynamic_task,out,'dynamic_test',key,args.workers)
    higher=run_jobs(higher_test,[(cfg,n,j) for n in cfg['higher_dimensions'] for j in range(cfg['higher_trials'])],out,'higher_dimension',key,args.workers)
    bounds=run_jobs(certificate_test,[(cfg,j) for j in range(cfg['certificate_trials'])],out,'certificate',key,args.workers)
    frames=dict(static_metrics=pd.DataFrame([row for group in static_groups for row in group['metrics']]),
        static_fisher=pd.DataFrame([row for group in static_groups for row in group['fisher']]),
        dynamic_metrics=pd.DataFrame([row for group in dynamic_groups for row in group['metrics']]),
        dynamic_bounds=pd.DataFrame([row for group in dynamic_groups for row in group['bounds']]),
        higher_dimension=pd.DataFrame([row for group in higher for row in group]),bounds=pd.DataFrame([row for group in bounds for row in group]))
    frames['paired_static']=pd.DataFrame(__import__('lib.contact_geometry_study',fromlist=['paired_metrics']).paired_metrics(frames['static_metrics'],['kind','t']))
    frames['paired_dynamic']=pd.DataFrame(__import__('lib.contact_geometry_study',fromlist=['paired_metrics']).paired_metrics(frames['dynamic_metrics'],['condition','t']))
    story,story_checks=build_stories(cfg,dynamic_selection);np.savez_compressed(out/'story.npz',**story)
    frames['story_checks']=pd.DataFrame(story_checks)
    for name,frame in frames.items():frame.to_csv(out/f'{name}.csv',index=False)
    if sources!={name:sha(ROOT/name) for name in SOURCES}:raise RuntimeError('Contact scientific sources changed during execution')
    static=frames['static_metrics'];dynamic=frames['dynamic_metrics'];higher=frames['higher_dimension'];allbounds=pd.concat([frames['bounds'],frames['dynamic_bounds']],ignore_index=True)
    checks=dict(seconds=perf_counter()-start,rows={name:len(frame) for name,frame in frames.items()},
        static_numerical_failures=int((~static.numerical_ok.fillna(False).astype(bool)).sum()),
        static_local_gaussian_unresolved=int((~static.loc[static.method.eq('laplace'),'local_ok'].fillna(False).astype(bool)).sum()),
        dynamic_numerical_failures=int((~dynamic.numerical_ok.fillna(False).astype(bool)).sum()),
        dynamic_local_gaussian_unresolved=int((~dynamic.loc[dynamic.method.eq('gaussian'),'local_ok'].fillna(False).astype(bool)).sum()),
        higher_numerical_failures=int((~higher.numerical_ok.fillna(False).astype(bool)).sum()),
        story_numerical_failures=int((~frames['story_checks'].numerical_ok.astype(bool)).sum()),
        bound_valid=int(allbounds.valid.sum()),bound_violations=int(allbounds.violated.sum()),
        maximum_modal_audit_error=float(max(static.modal_audit_error.max(),dynamic.modal_audit_error.max(),higher.modal_audit_error.max())),
        maximum_decision_identity_error=float(max(static.filter(regex='identity_error').max().max(),dynamic.filter(regex='identity_error').max().max())))
    write_findings(out,cfg,frames,checks)
    tables={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['manifest.json','config.json','status.json']}
    document=dict(declaration,status='completed',checks=checks,tables=tables)
    save(out/'manifest.json',document);save(out/'config.json',document);save(out/'status.json',dict(status='completed',source_hash=key,checks=checks))
    print(json.dumps(portable(checks),indent=2),flush=True)

if __name__=='__main__':main()

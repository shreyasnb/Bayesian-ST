"""Run serial, warmed cadence costs and independent numerical checks.

Paper timing requires all performance/prediction stages complete and an explicit
--isolated flag confirming the caller has stopped other experiment workers.
"""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse,hashlib,json,sys,platform,io,contextlib
from datetime import datetime,timezone
from time import perf_counter,process_time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import scipy
from lib.final_computation import (configuration,synthetic_case,CostCase,
    benchmark,numerical_accuracy,native,action,extract)
from lib.final_validation import require_gate,verify_component
from lib.models import increment
from lib.geometry import center

SOURCES=['lib/final_computation.py','scripts/run_final_computation.py','tests/test_final_computation.py',
    'lib/final_accuracy.py','tests/test_final_accuracy.py','lib/final_beamforming.py',
    'lib/final_contact.py','lib/final_prediction.py','scripts/run_final_prediction.py',
    'lib/causal_variance.py','lib/block_em.py','lib/final_theory.py','lib/models.py',
    'lib/geometry.py','lib/spectral_series.py','lib/beamforming_numerics.py',
    'lib/beamforming_decisions.py','lib/array_geometry_study.py','lib/contact_posterior.py','lib/contact_process_prediction.py',
    'benchmarks/common/great.py','pyproject.toml']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def portable(x):
    if isinstance(x,dict):return {str(k):portable(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [portable(v) for v in x]
    if isinstance(x,np.ndarray):return portable(x.tolist())
    if isinstance(x,np.generic):return portable(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def save(path,obj):path.write_text(json.dumps(portable(obj),indent=2,allow_nan=False)+'\n')


def task_cases(example,stage,cfg):
    """Exact deterministic performance trial0 inputs and frozen policies."""
    path=ROOT/'results'/example/'data'/stage/'performance/selected.json'
    if not path.exists():
        if stage=='paper':raise RuntimeError('Paper performance policies are required before isolated timing')
        return [],{},dict(status='not_evaluated',reason='Matching performance stage not yet complete; no policies substituted.')
    policies=json.loads(path.read_text());cases=[];arrays={}
    dependencies={str(path.relative_to(ROOT)):sha(path)}
    if example=='beamforming':
        from lib.final_beamforming import configuration as beam_config,beam_episode
        bc=beam_config(stage)
        for stationary,n,condition in [(True,32,'static'),(True,64,'static'),(True,128,'static'),(False,32,'smooth')]:
            selected=next(p for p in policies if p['n']==n and p['condition']==condition and p['stationary']==stationary)
            stream=beam_episode(bc,n,condition,0,'test',stationary)
            T=n if stationary else len(stream['U']);kind='task_stationary' if stationary else 'task_moving'
            h=stream['h'][:T];P=np.einsum('ti,tj->tij',h,h.conj())
            cases.append(CostCase(example,n,kind,stream['U'][:T],None,
                stream['sigma2'][:T],stream['tau2'][:T],stream['a'],
                stream['initial'][:,None],selected['shared_beta'],P,selected))
    else:
        from lib.final_contact import configuration as contact_config,generate_contact_trial
        stream=generate_contact_trial(contact_config(stage),'sphere',0,'test')
        selected=next(p for p in policies if p['condition']=='sphere');T=len(stream['U'])
        cases.append(CostCase(example,3,'task_contact',stream['U'],stream['V'],
            np.full(T,stream['sigma2']),None,stream['V'],np.eye(3)[:,1:],
            selected['beta'],stream['P'],selected,stream['G']))
    prediction_path=ROOT/'results'/example/'data'/stage/'prediction/selected.json'
    adaptive_status='not_evaluated: matching prediction policies not yet complete'
    if prediction_path.exists():
        from scripts.run_final_prediction import primary_stream
        from lib.final_prediction import configuration as prediction_config,tracker_kwargs
        pc=prediction_config(stage);selected=json.loads(prediction_path.read_text())['learned']
        condition='smooth' if example=='beamforming' else 'fast_sphere'
        stream=primary_stream(example,pc,condition,0,'test');T,n=stream['U'].shape
        kwargs=tracker_kwargs(example,n,selected,pc,learn=True)
        cases.append(CostCase(example,n,'task_adaptive',stream['U'],stream.get('V'),
            stream['sigma2'],stream['tau2'],stream['a'] if example=='beamforming' else stream['V'],
            np.eye(n,dtype=stream['U'].dtype),selected['beta'],stream['P'],
            dict(adaptive_policy=selected,adaptive_tracker_kwargs=kwargs),stream.get('G'),
            (kwargs['sigma2'],kwargs['tau2'])))
        dependencies[str(prediction_path.relative_to(ROOT))]=sha(prediction_path)
        adaptive_status='evaluated: exact causal trial0 '+condition+'; pre-correction action/normalized score every32; saved selected policy and initial variances'
    elif stage=='paper':raise RuntimeError('Paper prediction policies required for actual adaptive full-cycle timing')
    for case in cases:
        for name,value in dict(U=case.U,V=case.V,sigma2=case.sigma2,tau2=case.tau2,
            desired=case.desired,initial=case.initial,G=case.G,truth_for_scoring_only=case.truth).items():
            if value is not None:arrays[f'{case.case}_n{case.n}_{name}']=value
    return cases,arrays,dict(status='evaluated',dependencies=dependencies,episode=0,
        adaptive_status=adaptive_status,
        stationary='First n observations of the exact original 2n episode; endpoint output at T=n.',
        moving='Exact smooth n=32 episode; output every8 observations.',
        contact='Exact physical sphere n3 trial0; observed wrist G supplied to every tracker; response and soft-command action every4 observations.',
        generalized_scope='Other real dimensions are synthetic codimension-one fixtures, not physical contact performance.')


def primitives(example,cases,cfg):
    """Standalone operations are explanatory; never add these to cycle totals."""
    rows=[];rng=np.random.default_rng(cfg['seed']+99)
    for case in cases:
        n=case.n;cx=example=='beamforming';A=np.zeros((n,n),case.U.dtype)
        for j,u in enumerate(case.U):
            A=center(case.beta*A+increment(u,case.sigma2[j],
                None if case.V is not None else case.tau2[j],None if case.V is None else case.V[j],cx))
        Q=np.linalg.qr(case.initial if case.initial.shape==(n,n) else np.eye(n,dtype=A.dtype))[0]
        for spectrum in ('actual','span_80','span_400'):
            B=A if spectrum=='actual' else center((Q*np.linspace(0,float(spectrum.split('_')[1]),n))@Q.conj().T)
            if not cx and spectrum!='actual':B=-B
            M,_,_=native(B,example,cfg)
            ops=dict(correction=lambda:center(case.beta*B+increment(case.U[-1],case.sigma2[-1],None if case.V is not None else case.tau2[-1],None if case.V is None else case.V[-1],cx)),
                eigensystem=lambda:np.linalg.eigh(B),
                mean=lambda:native(B,example,cfg),
                action=lambda:action(M,example,case.desired,case.sigma2[-1],None if not cx else case.tau2[-1]),
                transport=lambda:center(Q@B@Q.conj().T))
            for f in ops.values():
                try:f()
                except (ArithmeticError,np.linalg.LinAlgError):pass
            for rep in range(cfg['repetitions']):
                for op in rng.permutation(list(ops)):
                    tic=perf_counter();cpu=process_time();failure=''
                    ok=True
                    try:
                        result=ops[op]()
                        if op=='mean':ok=bool(result[2]['numerical_ok'])
                    except (ArithmeticError,np.linalg.LinAlgError) as error:failure=str(error)
                    rows.append(dict(example=example,n=n,case=case.case,spectrum=spectrum,
                        span=float(np.ptp(np.linalg.eigvalsh(B))),operation=op,repetition=rep,
                        seconds=perf_counter()-tic,cpu_seconds=process_time()-cpu,
                        numerical_ok=bool(not failure and ok),failure=failure,
                        scope='Standalone operation, not added to the full-cycle timings. Mean includes eigensystem.'))
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['verify','pilot','paper'],default='verify')
    parser.add_argument('--example',choices=['beamforming','contact','both'],default='both')
    parser.add_argument('--isolated',action='store_true')
    args=parser.parse_args();cfg=configuration(args.stage)
    if args.stage=='paper' and not args.isolated:
        parser.error('Paper timing must be serial on an idle experiment host: pass --isolated after all worker pools finish')
    gate=require_gate(ROOT) if args.stage=='paper' else None
    if args.stage=='paper':
        for ex in ('beamforming','contact'):
            for component in ('performance','prediction','theory'):verify_component(ROOT,ex,'paper',component)
    source_hashes={name:sha(ROOT/name) for name in SOURCES}
    captured=io.StringIO()
    with contextlib.redirect_stdout(captured):np.show_config()
    hardware=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        platform=platform.platform(),machine=platform.machine(),processor=platform.processor(),
        blas_environment={key:os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        numpy_configuration=captured.getvalue(),isolated_asserted=args.isolated,
        scope='Container CPU environment; not a hardware worst-case timing guarantee.')
    for example in (('beamforming','contact') if args.example=='both' else (args.example,)):
        cases_task,task_arrays,task_meta=task_cases(example,args.stage,cfg)
        sources=dict(source_hashes,**task_meta.get('dependencies',{}))
        out=ROOT/'results'/example/'data'/args.stage/'computation'
        tables=ROOT/'results'/example/'tables'/args.stage/'computation'
        out.mkdir(parents=True,exist_ok=True);tables.mkdir(parents=True,exist_ok=True)
        existing=out/'manifest.json'
        if existing.exists():
            meta=verify_component(ROOT,example,args.stage,'computation')
            if meta['config']!=cfg:raise RuntimeError('Preserve obsolete computation configuration before rerunning')
            print('Verified cached',example,args.stage,flush=True);continue
        save(out/'preregistered.json',dict(config=cfg,sources=sources,
            registered_utc=datetime.now(timezone.utc).isoformat(),hardware=hardware))
        start=perf_counter();cases=[synthetic_case(example,n,cfg) for n in cfg['dimensions']]
        cases+=cases_task
        rows,memory,parameters=benchmark(example,cases,cfg,lambda m:print(m,flush=True))
        frame=pd.DataFrame(rows);frame.to_csv(tables/'pipeline_timings.csv',index=False)
        keys=['example','n','case','method','action_cadence','report_every','learn_every','em_iterations','observations']
        summary=frame.groupby(keys,dropna=False).agg(
            repetitions=('repetition','count'),median_ms=('wall_seconds',lambda x:1000*x.median()),
            p90_ms=('wall_seconds',lambda x:1000*x.quantile(.9)),
            correction_ms=('correction_seconds',lambda x:1000*x.median()),
            action_ms=('action_seconds',lambda x:1000*x.median()),
            report_ms=('report_seconds',lambda x:1000*x.median()),
            fitting_ms=('fitting_seconds',lambda x:1000*x.median()),
            overhead_ms=('overhead_seconds',lambda x:1000*x.median()),
            mean_ok=('mean_ok','all'),fit_ok=('fit_ok','all'),report_ok=('report_ok','all'),
            task_loss=('task_loss','mean')).reset_index()
        summary.to_csv(tables/'summary.csv',index=False);pd.DataFrame(memory).to_csv(tables/'memory.csv',index=False)
        accuracy,accuracy_parameters=numerical_accuracy(example,cases,cfg,parameters)
        reports=frame[frame.report_every.gt(0)]
        for _,r in reports.drop_duplicates(['n','case','method']).iterrows():
            accuracy.append(dict(example=example,n=r.n,case='scheduled_reports',
                mean_ok=r.mean_ok,normalizer_ok=None,second_moment_evaluated=False,
                second_moment_ok=None,tail_evaluated=True,tail_ok=r.report_tail_ok,
                state_report_ok=r.report_state_ok,tail_requested_relative_mcse=cfg['report_relative_mcse'] if example=='beamforming' else np.nan,
                tail_precision_scope='Independent iid threshold/check batches' if example=='beamforming' else 'Continuous CDF with doubled angular-resolution check',
                reference_scope='Separate report precision flags; not inferred from native first-moment truncation.'))
        pd.DataFrame(accuracy).to_csv(tables/'accuracy.csv',index=False)
        primitives(example,[c for c in cases if c.case=='data_T64'],cfg).to_csv(tables/'primitives.csv',index=False)
        np.savez_compressed(out/'parameters.npz',**parameters,**{'accuracy_'+k:v for k,v in accuracy_parameters.items()})
        np.savez_compressed(out/'task_inputs.npz',**task_arrays)
        save(out/'task_matching.json',task_meta)
        save(out/'hardware.json',hardware)
        checks=dict(pipeline_targets=len(summary),timed_cycles=len(frame),
            numerical_failures=int((~frame.mean_ok.astype(bool)).sum()),
            fit_failures=int((~frame.fit_ok.astype(bool)).sum()),
            report_failures=int((~frame.report_ok.astype(bool)).sum()),
            predictive_score_failures=int((~frame.predictive_score_ok.fillna(False).astype(bool)).sum()),
            maximum_phase_excess_seconds=float(np.maximum(0,-frame.overhead_seconds).max()),
            independent_accuracy_unresolved=sum(not bool(r.get('mean_ok',False)) for r in accuracy if r.get('case')!='scheduled_reports'),
            elapsed_seconds=perf_counter()-start)
        save(out/'checks.json',checks)
        if any(sha(ROOT/name)!=value for name,value in sources.items()):raise RuntimeError('Scientific sources changed during timing')
        artifacts={str(path.relative_to(ROOT)):sha(path) for folder in (out,tables)
            for path in sorted(folder.glob('*')) if path.is_file() and path.name!='manifest.json'}
        save(existing,dict(status='completed',example=example,stage=args.stage,component='computation',
            config=cfg,sources=sources,artifacts=artifacts,checks=checks,
            timing_setting=hardware,task_matching=task_meta,
            final_science_hash=gate.get('final_science_hash') if gate else None))
        print(example,json.dumps(checks),flush=True)


if __name__=='__main__':main()

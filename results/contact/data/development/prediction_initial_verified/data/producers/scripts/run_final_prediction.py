"""Verify, pilot and freeze future-only causal variance/prediction studies."""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from datetime import datetime, timezone
import hashlib,json,gzip,sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from lib.final_prediction import (configuration,seed,adaptation_trial,calibration_trial,
    interval,binomial_interval)
from lib.causal_variance import CausalVarianceTracker

SOURCES=['lib/causal_variance.py','lib/final_prediction.py','scripts/run_final_prediction.py',
    'tests/test_causal_variance.py','lib/block_em.py','lib/models.py','lib/geometry.py',
    'lib/spectral_series.py','lib/guide_reliability.py','lib/contact_posterior.py',
    'lib/beamforming_numerics.py','lib/contact_mechanics.py','lib/final_beamforming.py',
    'lib/final_contact.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def portable(x):
    if isinstance(x,dict):return {str(k):portable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [portable(v) for v in x]
    if isinstance(x,np.ndarray):return portable(x.tolist())
    if isinstance(x,np.generic):return portable(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def write(path,x):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(portable(x),indent=2,allow_nan=False)+'\n');temporary.replace(path)
def digest(value):return hashlib.sha256(json.dumps(portable(value),sort_keys=True).encode()).hexdigest()


def primary_stream(example,cfg,condition,episode,split):
    if example=='beamforming':
        from lib.final_beamforming import generate_stream
        base=cfg['validation_seed'] if split=='validation' else cfg['test_seed']
        entropy=np.random.SeedSequence([base+cfg['seed_offset'],32,episode,31]).generate_state(1)[0]
        d=generate_stream(cfg['primary_beam_n'],cfg['primary_beam_T'],condition,int(entropy),
                          sigma2=.2,tau2=1.,overlap=.5,dt=cfg['primary_beam_dt'])
        d['P']=np.einsum('ti,tj->tij',d['h'],d['h'].conj())
        d['dt']=cfg['primary_beam_dt'];d['G']=None
        return d
    from lib.final_contact import configuration as contact_configuration,generate_contact_trial
    own=contact_configuration(cfg['stage'])
    own.update(T=cfg['primary_contact_T'],dt=cfg['primary_contact_dt'])
    condition_name='plane' if condition=='noise_change' else 'fast_sphere'
    d=generate_contact_trial(own,condition_name,episode,split)
    n=len(d['U']);sigma2=np.full(n,.004**2)
    if condition=='noise_change':sigma2[n//2:]*=4
    d['U']=d['clean']+np.sqrt(sigma2)[:,None]*d['standard_noise']
    d['sigma2']=sigma2;d['tau2']=None;d['dt']=cfg['primary_contact_dt'];d['query_scale']=.02
    return d


def worker(arguments):
    kind,example,cfg,episode,split,condition,policy,method=arguments
    try:
        if kind=='adaptation':
            stream=primary_stream(example,cfg,condition,episode,split)
            policies={name:policy[name] for name in policy}
            return dict(kind=kind,rows=adaptation_trial(example,stream,policies,cfg,episode,
                split,condition,story=split=='test' and episode==0,
                method_subset=None if method is None else [method]))
        regime=condition if kind=='calibration' else 'fixed_state'
        stress='matched' if kind=='calibration' else condition
        return dict(kind=kind,rows=calibration_trial(example,regime,cfg,episode,split,policy,stress,kind=='mismatch'))
    except (ArithmeticError,ValueError,np.linalg.LinAlgError) as error:
        return dict(kind='failure',example=example,episode=episode,split=split,
            condition=condition,error=type(error).__name__+': '+str(error))


def execute(tasks,out,phase,sources,workers):
    directory=out/'checkpoints'/phase;directory.mkdir(parents=True,exist_ok=True)
    producer=digest(sources);done={};pending=[]
    for i,task in enumerate(tasks):
        p=directory/f'{i:05d}.json'
        if p.exists():
            old=json.loads(p.read_text())
            if old['producer']==producer and old['task']==digest(task) and old['result_hash']==digest(old['result']):
                arrays={}
                if old.get('array_file'):
                    a=out/old['array_file']
                    if sha(a)!=old['array_sha']:raise RuntimeError('Changed preassigned story')
                    with np.load(a,allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
                result=old['result']
                if result['kind']=='adaptation':result['rows'][2]=arrays
                done[i]=result;continue
        pending.append((i,task,p))
    def retain(i,result,p):
        arrays={}
        if result.get('kind')=='adaptation':
            result['rows']=list(result['rows'])
            arrays=result['rows'][2];result['rows'][2]={}
        meta=dict(producer=producer,task=digest(tasks[i]))
        if arrays:
            a=out/'stories'/f'{phase}_{i:05d}.npz';a.parent.mkdir(exist_ok=True)
            np.savez_compressed(a,**arrays);meta.update(array_file=str(a.relative_to(out)),array_sha=sha(a))
        result=portable(result)
        meta.update(result=result,result_hash=digest(result));write(p,meta)
        if result['kind']=='adaptation':result['rows'][2]=arrays
        done[i]=result
        if len(done)%32==0 or len(done)==len(tasks):print(phase,len(done),'/',len(tasks),flush=True)
    if workers==1:
        for i,task,p in pending:retain(i,worker(task),p)
    elif pending:
        with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
            futures={pool.submit(worker,task):(i,p) for i,task,p in pending}
            for future in as_completed(futures):
                i,p=futures[future];retain(i,future.result(),p)
    return [done[i] for i in range(len(tasks))]


def validation(example,cfg,out,sources,workers):
    conditions=cfg['beam_conditions'] if example=='beamforming' else cfg['contact_conditions']
    tasks=[];meta=[]
    for method in ['known','misspecified','learned']:
        candidates=cfg['policy_candidates'] if method=='learned' else cfg['policy_candidates'][:3]
        for index,policy in enumerate(candidates):
            for condition in conditions:
                for episode in range(cfg['validation_episodes']):
                    tasks.append(('adaptation',example,cfg,episode,'validation',condition,{method:policy},method))
                    meta.append((method,index,condition,episode))
    results=execute(tasks,out,'validation',sources,workers)
    rows=[];failures=[]
    for (method,index,condition,episode),result in zip(meta,results):
        if result['kind']=='failure':failures.append(result);continue
        metrics=pd.DataFrame(result['rows'][0]);metrics=metrics[metrics.control.eq('mean')]
        values=metrics.loc[metrics.t>=32,'true_task_loss'].to_numpy()
        rows.append(dict(method=method,candidate=index,condition=condition,episode=episode,
            loss=float(np.mean(values)),numerical_ok=bool(np.isfinite(values).all() and metrics.mean_ok.all()),
            beta=cfg['policy_candidates'][index]['beta'],
            block_length=cfg['policy_candidates'][index]['block_length'],
            em_iterations=cfg['policy_candidates'][index]['em_iterations']))
    frame=pd.DataFrame(rows);selected={}
    for method in ['known','misspecified','learned']:
        g=frame[frame.method.eq(method)]
        score=g.groupby('candidate').loss.mean()
        eligibility=g.groupby('candidate').numerical_ok.all()
        eligible=score[eligibility]
        if eligible.empty:raise ArithmeticError('No fully resolved validation candidate for '+method)
        chosen=int(eligible.idxmin());selected[method]=cfg['policy_candidates'][chosen]
    return frame,selected,failures


def summaries(adaptation,calibration,scores):
    output={}
    if not adaptation.empty:
        agg=adaptation.groupby(['condition','episode','method','control'],dropna=False).agg(
            future_task_loss=('future_task_loss','mean'),true_task_loss=('true_task_loss','mean'),
            output_oracle_db=('output_oracle_db','mean'),
            heldout_output_oracle_db=('heldout_output_oracle_db','mean'),
            prequential_score=('prequential_score','mean'),response_mse=('response_mse','mean'),
            command_cost=('command_cost','mean')).reset_index()
        rows=[]
        for keys,g in agg.groupby(['condition','method','control'],sort=False):
            common=dict(zip(['condition','method','control'],keys),episodes=g.episode.nunique())
            for endpoint in ['future_task_loss','true_task_loss','output_oracle_db',
                             'heldout_output_oracle_db','prequential_score','response_mse','command_cost']:
                val,lo,hi=interval(g[endpoint])
                rows.append(dict(common,endpoint=endpoint,mean=val,lower=lo,upper=hi))
        output['adaptation_summary']=pd.DataFrame(rows)
        pairs=[]
        for condition,g in agg.groupby('condition',sort=False):
            base=g[g.method.eq('learned')&g.control.eq('mean')].set_index('episode')
            for (method,control),h in g.groupby(['method','control'],sort=False):
                if method=='learned' and control=='mean':continue
                h=h.set_index('episode');idx=base.index.intersection(h.index)
                for endpoint in ['future_task_loss','true_task_loss','output_oracle_db','response_mse','command_cost']:
                    delta=h.loc[idx,endpoint]-base.loc[idx,endpoint]
                    val,lo,hi=interval(delta)
                    pairs.append(dict(condition=condition,reference=method,reference_control=control,
                        endpoint=endpoint,episodes=len(idx),gain=val,lower=lo,upper=hi,
                        interval='Pointwise paired Student95% across independent episodes; time/query means formed inside episodes'))
        output['paired']=pd.DataFrame(pairs)
    if not calibration.empty:
        rows=[]
        keys=['example','regime','stress','split','method','t','kind','nominal']
        for values,g in calibration.groupby(keys,sort=False,dropna=False):
            count=len(g);success=int(g.covered.sum());lo,hi=binomial_interval(success,count)
            width,wl,wu=interval(g.width)
            rows.append(dict(zip(keys,values),episodes=count,coverage=success/count,lower=lo,upper=hi,
                width=width,width_lower=wl,width_upper=wu,unresolved=int((~g.numerical_ok).sum()),
                interval='Pointwise exact binomial95%; one fixed checkpoint, independent episodes'))
        output['calibration_summary']=pd.DataFrame(rows)
    if not scores.empty:
        rows=[]
        for values,g in scores.groupby(['example','regime','stress','split','method'],sort=False):
            ep=g.groupby('episode').mean(numeric_only=True)
            gain,lo,hi=interval(ep.mixture_log_score-ep.gaussian_log_score)
            risk,rl,rh=interval(ep.risk_realized-ep.risk_predicted)
            row=dict(zip(['example','regime','stress','split','method'],values),
                episodes=len(ep),mixture_score=float(ep.mixture_log_score.mean()),
                gaussian_score=float(ep.gaussian_log_score.mean()),score_gain=gain,
                score_lower=lo,score_upper=hi,risk_difference=risk,risk_lower=rl,risk_upper=rh,
                unresolved=int((~g.numerical_ok).sum()))
            for control in ['mean','mode','iso']:
                row['response_loss_'+control]=float(ep['response_loss_'+control].mean())
            rows.append(row)
        output['score_summary']=pd.DataFrame(rows)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['verify','pilot','paper'],default='verify')
    parser.add_argument('--example',choices=['beamforming','contact','both'],default='both')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--section',choices=['all','adaptation','prediction'],default='all')
    args=parser.parse_args();cfg=configuration(args.stage)
    cfg['section']=args.section
    if args.stage=='paper' and args.section!='all':
        raise ValueError('The paper component requires all declared adaptation/prediction sections')
    if args.stage=='paper':
        from lib.final_validation import require_gate
        require_gate(ROOT)
    sources={name:sha(ROOT/name) for name in SOURCES}
    examples=['beamforming','contact'] if args.example=='both' else [args.example]
    for example in examples:
        start=perf_counter()
        out=ROOT/'results'/example/'data'/args.stage/'prediction'
        table=ROOT/'results'/example/'tables'/args.stage/'prediction'
        out.mkdir(parents=True,exist_ok=True);table.mkdir(parents=True,exist_ok=True)
        if (out/'manifest.json').exists():
            old=json.loads((out/'manifest.json').read_text())
            if old['config']!=cfg or old['sources']!=sources:raise RuntimeError('Preserve prior outputs before changing scientific settings')
            for name,value in old['artifacts'].items():
                if sha(ROOT/name)!=value:raise RuntimeError('Changed cached artifact '+name)
            print('Verified completed',example,args.stage,'prediction');continue
        registration=out/'preregistered.json'
        if registration.exists():
            previous=json.loads(registration.read_text())
            if previous['config']!=cfg or previous['sources']!=sources:
                raise RuntimeError('Preserve the previous incomplete profile and its source snapshots before rerunning changed science')
        else:
            write(registration,dict(config=cfg,sources=sources,registered_utc=datetime.now(timezone.utc).isoformat()))
            for name in SOURCES:
                target=out/'producers'/name;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes((ROOT/name).read_bytes())
        failures=[];selected={name:cfg['policy_candidates'][1] for name in ['known','misspecified','learned']}
        if args.section in ('all','adaptation'):
            val,selected,failed=validation(example,cfg,out,sources,args.workers)
            val.to_csv(table/'validation.csv',index=False);failures+=failed
        write(out/'selected.json',selected)
        tasks=[]
        if args.section in ('all','adaptation'):
            conditions=cfg['beam_conditions'] if example=='beamforming' else cfg['contact_conditions']
            tasks.extend(('adaptation',example,cfg,ep,'test',c,selected,None)
                for c in conditions for ep in range(cfg['test_episodes']))
        if args.section in ('all','prediction'):
            p=selected['learned']
            for split,count in [('validation',cfg['calibration_validation_episodes']),('test',cfg['calibration_test_episodes'])]:
                tasks.extend(('calibration',example,cfg,ep,split,regime,p,None)
                    for regime in ['prior_predictive','fixed_state','drifting'] for ep in range(count))
            stresses=(['matched','desired_contamination','additional_interference','correlated_noise','fast_drift']
                if example=='beamforming' else ['matched','known_resistance_correct_noise',
                    'known_resistance_raw_isotropic','wrong_resistance','friction','contact_loss','position_differences'])
            tasks.extend(('mismatch',example,cfg,ep,'test',stress,p,None)
                for stress in stresses for ep in range(cfg['mismatch_episodes']))
        results=execute(tasks,out,'test',sources,args.workers)
        adap=[];calib=[];scores=[];events=[];stories={}
        for item in results:
            if item['kind']=='failure':failures.append(item);continue
            a,b,c=item['rows']
            if item['kind']=='adaptation':
                adap+=a;events+=b
                for key,value in c.items():stories[key]=value
            else:calib+=a;scores+=b;events+=c
        frames=dict(adaptation=pd.DataFrame(adap),calibration=pd.DataFrame(calib),scores=pd.DataFrame(scores))
        for name,frame in frames.items():frame.to_csv(table/(name+'.csv'),index=False)
        if len(scores):
            frames['scores'][frames['scores'].cohort.eq('mismatch')].to_csv(table/'mismatch.csv',index=False)
        for name,frame in summaries(**frames).items():frame.to_csv(table/(name+'.csv'),index=False)
        with gzip.open(out/'fit_events.json.gz','wt') as f:json.dump(portable(events),f,allow_nan=False)
        if stories:np.savez_compressed(out/'stories.npz',**stories)
        write(out/'failures.json',failures)
        checks=dict(requested_jobs=len(tasks),failed_jobs=len(failures),
            adaptation_rows=len(adap),calibration_rows=len(calib),score_rows=len(scores),
            unresolved_adaptation_rows=int((~frames['adaptation'].mean_ok).sum()) if len(adap) else 0,
            unresolved_calibration_rows=int((~frames['calibration'].numerical_ok).sum()) if len(calib) else 0,
            unresolved_score_rows=int((~frames['scores'].numerical_ok).sum()) if len(scores) else 0,
            maximum_state_change_from_fit=max([e.get('state_change_from_fit',0) for e in events] or [0]),
            fits=len(events),elapsed_seconds=perf_counter()-start,
            causal_model='Saved preceding prior; completed disjoint blocks; fitfuture-only; streamingAunchanged',
            probability_scope='Exact prior-predictive identity tested only beta1 knownscales; actual forgetting/learning inclusion is empirical')
        write(out/'checks.json',checks)
        if any(sha(ROOT/name)!=value for name,value in sources.items()):raise RuntimeError('Science sources changed during execution')
        files=[p for folder in [out,table] for p in folder.rglob('*') if p.is_file() and 'checkpoints' not in p.parts and p.name!='manifest.json']
        write(out/'manifest.json',dict(status='completed',example=example,stage=args.stage,
            component='prediction',config=cfg,sources=sources,
            artifacts={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},checks=checks))
        print(json.dumps(portable(checks),indent=2),flush=True)


if __name__=='__main__':main()

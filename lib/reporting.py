"""Content-addressed experiment tables and small common plotting routines."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .validation import require_gate


FIELD_FUNCTIONS = ('_rng','_posterior','_credible','_score_prior','_field_data',
                   '_tune_field','_noise_study','run_case2')


def field_source(path):
    """Numerical dependency closure, excluding the independent array driver."""
    import ast
    text=Path(path).read_text();tree=ast.parse(text)
    nodes={n.name:ast.get_source_segment(text,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    return '\n\n'.join(nodes[n] for n in FIELD_FUNCTIONS)


def config_digest(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()


def simulation_hash(root,case,config):
    root=Path(root);gate=require_gate(root)
    h=hashlib.sha256((gate['core_hash']+json.dumps(config,sort_keys=True)).encode())
    source=field_source(root/'lib/experiments.py') if case=='case2' else (root/'lib/experiments.py').read_text()
    h.update(source.encode())
    return h.hexdigest()


def run_or_load(root,case,config,refresh=False):
    from .experiments import run_case1,run_case2
    root=Path(root);out=root/'results';key=simulation_hash(root,case,config)
    meta=out/f'{case}_config.json'
    if meta.exists() and not refresh:
        saved=json.loads(meta.read_text())
        if saved['source_hash']==key:
            if saved.get('requested_config')!=config or saved.get('config_digest')!=config_digest(saved['config']):
                raise RuntimeError('Cached configuration changed; rerun with refresh=True')
            for name,h in saved['tables'].items():
                if hashlib.sha256((out/f'{case}_{name}.csv').read_bytes()).hexdigest()!=h:
                    raise RuntimeError('Cached simulation table changed; rerun with refresh=True')
            print(f'{case}: loading verified simulation tables')
            return {**{n:pd.read_csv(out/f'{case}_{n}.csv') for n in saved['tables']},'config':saved['config']}
    print(f'{case}: simulating with {config}',flush=True)
    result=({'case1':run_case1,'case2':run_case2}[case])(**config)
    tables={}
    for name,frame in result.items():
        if isinstance(frame,pd.DataFrame):
            p=out/f'{case}_{name}.csv';frame.to_csv(p,index=False)
            tables[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    meta.write_text(json.dumps(dict(source_hash=key,requested_config=config,config=result['config'],
                                   config_digest=config_digest(result['config']),tables=tables),indent=2))
    return result


def wilson(successes,total,z=1.96):
    p=successes/total;den=1+z*z/total
    mid=(p+z*z/(2*total))/den
    half=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return min(p,max(0.,mid-half)),max(p,min(1.,mid+half))


def coverage_table(frame,group='scenario'):
    rows=[]
    for label,g in frame.groupby(group):
        for kind in ['ball','hpd']:
            values=g[f'{kind}_covered'].dropna().astype(bool)
            low,high=wilson(int(values.sum()),len(values))
            rows.append(dict(scenario=label,region=kind.upper() if kind=='hpd' else 'Ball',
                             coverage=float(values.mean()),low=low,high=high,n=len(values)))
    return pd.DataFrame(rows)


def error_curve(ax,frame,value,methods):
    for method in methods:
        g=frame[frame.method==method].groupby('t')[value].agg(['mean','std','count'])
        sem=g['std']/np.sqrt(g['count'])
        ax.errorbar(g.index,g['mean'],yerr=sem,marker='.',capsize=2,label=method)
    ax.set_xlabel('Observations')

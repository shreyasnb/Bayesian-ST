"""Compute the native-complex array prediction figures or load verified tables."""
import os
for v in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[v]='1'
from pathlib import Path
import sys,json,hashlib,inspect,platform
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,scipy
from lib.validation import require_gate
from lib.array_story import run_array_story
from lib import moments


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    gate=require_gate(ROOT);out=ROOT/'results'
    files={p:sha(ROOT/p) for p in ['lib/array_story.py','lib/experiments.py','scripts/run_array_story.py','lib/filter.py','lib/models.py','lib/geometry.py','benchmarks/case1/baselines.py']}
    functions={name:hashlib.sha256(inspect.getsource(getattr(moments,name)).encode()).hexdigest() for name in ['sample_bingham','chain_diagnostics','_batch_mcse','effective_sample_size','MomentResult','_hermitian']}
    inputs={p:sha(out/p) for p in ['case1_metrics.csv','case1_tuning.csv']}
    versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
    key=hashlib.sha256(json.dumps(dict(files=files,functions=functions,inputs=inputs,versions=versions),sort_keys=True).encode()).hexdigest()
    meta=out/'array_story_config.json'
    if meta.exists():
        saved=json.loads(meta.read_text())
        if saved['source_hash']==key and all(sha(out/f'array_story_{n}.csv')==h for n,h in saved['tables'].items()):
            print('Array uncertainty: verified cache');return
    result=run_array_story(pd.read_csv(out/'case1_metrics.csv'),pd.read_csv(out/'case1_tuning.csv'))
    assert files=={p:sha(ROOT/p) for p in files},'Source changed during computation'
    tables={}
    for name,frame in result.items():
        if isinstance(frame,pd.DataFrame):
            p=out/f'array_story_{name}.csv';frame.to_csv(p,index=False);tables[name]=sha(p)
    meta.write_text(json.dumps(dict(source_hash=key,files=files,functions=functions,inputs=inputs,versions=versions,
        config=result['config'],tables=tables,core_hash=gate['core_hash']),indent=2))


if __name__=='__main__':main()

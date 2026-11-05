"""Revalidate historical simulations after an unused convenience-API extension.

This does not claim a rerun. It requires the old numerical experiment, filtering,
and sampling code paths to be unchanged. Old provenance is retained explicitly.
"""
from pathlib import Path
import argparse,ast,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from lib.validation import require_gate
from lib.reporting import simulation_hash,field_source


def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha(path):return sha_bytes(Path(path).read_bytes())
def old_blob(commit,name):return subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=ROOT)
def definitions(source):
    tree=ast.parse(source)
    return {n.name:ast.get_source_segment(source,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--from-commit',required=True);args=p.parse_args()
    old_gate=json.loads(old_blob(args.from_commit,'results/validation.json'))
    new_gate=require_gate(ROOT);old_core=old_gate['core_hash'];new_core=new_gate['core_hash']
    if old_core==new_core:raise RuntimeError('No core revision to revalidate')
    unchanged=['lib/filter.py','lib/models.py','lib/geometry.py','lib/em.py','lib/theory.py','lib/experiments.py',
               'lib/decision_study.py','scripts/run_decision_study.py','lib/reporting.py','scripts/check_array_accuracy.py']
    unchanged+=subprocess.check_output(['git','ls-tree','-r','--name-only',args.from_commit,'benchmarks'],cwd=ROOT,text=True).splitlines()
    for name in unchanged:assert (ROOT/name).read_bytes()==old_blob(args.from_commit,name),name
    old_moments=definitions(old_blob(args.from_commit,'lib/moments.py').decode())
    new_moments=definitions((ROOT/'lib/moments.py').read_text())
    retained={name:sha_bytes(source.encode()) for name,source in old_moments.items() if name!='mean_bingham'}
    for name,digest in retained.items():assert sha_bytes(new_moments[name].encode())==digest,name
    proof=dict(from_commit=args.from_commit,original_core_hash=old_core,revalidated_core_hash=new_core,
        unchanged_files={name:sha(ROOT/name) for name in unchanged},unchanged_moment_definitions=retained,
        reason='Only mean_bingham convenience dispatch and a new normalizer API changed; archived simulations call the unchanged sampler or their own unchanged quadrature.',
        rerun=False)
    out=ROOT/'results'
    for case in ['case1','case2']:
        path=out/f'{case}_config.json';meta=json.loads(path.read_text())
        old_meta=json.loads(old_blob(args.from_commit,f'results/{case}_config.json'))
        assert meta['source_hash']==old_meta['source_hash'],'Already migrated or unknown provenance'
        for name,h in meta['tables'].items():assert sha(out/f'{case}_{name}.csv')==h
        meta['original_source_hash']=meta.get('original_source_hash',meta['source_hash'])
        meta['source_hash']=simulation_hash(ROOT,case,meta['requested_config'])
        meta.setdefault('revalidations',[]).append(proof)
        path.write_text(json.dumps(meta,indent=2))
    path=out/'case1_accuracy.json';meta=json.loads(path.read_text())
    assert meta['core_hash']==old_core
    assert sha(out/'case1_accuracy.csv')==meta['result_sha256']
    meta['original_core_hash']=old_core;meta['original_source']=meta['source']
    meta['original_input_sha256']=meta['input_sha256']
    updated=json.loads(json.dumps(meta['source']))
    for name in ['lib/moments.py','lib/validation.py']:updated['files'][name]=sha(ROOT/name)
    updated['sha256']=sha_bytes(json.dumps(updated['files'],sort_keys=True).encode())
    meta['source']=updated;meta['core_hash']=new_core
    meta['input_sha256']={name:sha(ROOT/name) for name in meta['input_sha256']}
    meta.setdefault('revalidations',[]).append(proof)
    path.write_text(json.dumps(meta,indent=2))
    path=out/'decision_study_config.json';meta=json.loads(path.read_text())
    assert meta['core_hash']==old_core
    for name,h in meta['sources'].items():assert sha(ROOT/name)==h
    for name,h in meta['tables'].items():assert sha(out/f'decision_study_{name}.csv')==h
    payload=dict(config=meta['requested_config'],sources=meta['sources'],versions=meta['versions'],core_hash=old_core)
    assert sha_bytes(json.dumps(payload,sort_keys=True).encode())==meta['source_hash']
    meta['original_core_hash']=old_core;meta['original_source_hash']=meta['source_hash']
    payload['core_hash']=new_core;meta['source_hash']=sha_bytes(json.dumps(payload,sort_keys=True).encode());meta['core_hash']=new_core
    meta.setdefault('revalidations',[]).append(proof);path.write_text(json.dumps(meta,indent=2))
    (out/'revision_provenance.json').write_text(json.dumps(proof,indent=2))
    print('Historical data unchanged; numerical dependency paths verified against',args.from_commit)


if __name__=='__main__':main()

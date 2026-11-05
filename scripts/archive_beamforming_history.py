"""Pin old results to their actual producer dependencies and general certificate.

No old simulation or cache gate is bypassed. Historical figures are displayed
read-only after artifact checks; the original numerical tree matching their
certificate is included for reproducibility independently of new tests.
"""
from pathlib import Path
import ast,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE='3b47f8d7765d125fff6a1bbe8c8e9d43a8627517'


def blob(revision,name):
    return subprocess.check_output(['git','show',f'{revision}:{name}'],cwd=ROOT,stderr=subprocess.DEVNULL)


def sha(data):return hashlib.sha256(data).hexdigest()


def tree_names(revision,*paths):
    return subprocess.check_output(['git','ls-tree','-r','--name-only',revision,*paths],cwd=ROOT,text=True).splitlines()


def core_paths(revision):
    source=blob(revision,'lib/validation.py').decode();tree=ast.parse(source)
    core=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='CORE' for t in n.targets))
    return core+sorted(p for p in tree_names(revision,'benchmarks','tests') if p.endswith('.py'))


def main():
    out=ROOT/'results/beamforming_unified/archive';out.mkdir(parents=True,exist_ok=True)
    gate=json.loads(blob(BASE,'results/validation.json'))
    revisions=subprocess.check_output(['git','rev-list','--all'],cwd=ROOT,text=True).splitlines()
    revision=None
    for candidate in revisions:
        try:
            paths=core_paths(candidate);digest=hashlib.sha256()
            for name in paths:digest.update(name.encode());digest.update(blob(candidate,name))
        except (subprocess.CalledProcessError,StopIteration):continue
        if digest.hexdigest()==gate['core_hash']:revision=candidate;break
    if revision is None:raise RuntimeError('Cannot reconstruct the historical general certificate')
    paths=core_paths(revision);snapshots={}
    for name in tree_names(revision,'lib','benchmarks','tests','scripts'):
        if not name.endswith('.py'):continue
        data=blob(revision,name);relative='certificate_source/'+name
        p=out/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        snapshots[relative]=sha(data)
    metadata_names=['case1_config.json','case1_accuracy.json','array_story_config.json',
        'array_envelope_config.json','decision_study_config.json']
    producer_checks=[]
    for metadata in metadata_names:
        value=json.loads((ROOT/'results'/metadata).read_text())
        sources=value.get('files',value.get('source',{}).get('files',value.get('sources',{})))
        if not isinstance(sources,dict):raise RuntimeError(f'Unexpected source declaration in {metadata}')
        for name,expected in sources.items():
            if not isinstance(expected,str) or len(expected)!=64:continue
            resolved=None;data=None
            for candidate in [revision,*revisions]:
                try:content=blob(candidate,name)
                except subprocess.CalledProcessError:continue
                if sha(content)==expected:resolved=candidate;data=content;break
            if resolved is None:raise RuntimeError(f'Historical producer source unavailable: {metadata}, {name}')
            relative='producer_source/'+expected[:12]+'/'+name
            p=out/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
            snapshots[relative]=expected
            producer_checks.append(dict(metadata=metadata,path=name,sha256=expected,
                revision=resolved,snapshot=relative,current_matches=sha((ROOT/name).read_bytes())==expected))
        if metadata=='array_envelope_config.json':
            assert sha((ROOT/'results/array_envelope_bounds.csv').read_bytes())==value['table_sha256']
        elif metadata=='case1_accuracy.json':
            assert sha((ROOT/'results/case1_accuracy.csv').read_bytes())==value['result_sha256']
        else:
            stem={'case1_config.json':'case1','array_story_config.json':'array_story',
                  'decision_study_config.json':'decision_study'}[metadata]
            for name,expected in value['tables'].items():
                assert sha((ROOT/'results'/f'{stem}_{name}.csv').read_bytes())==expected,name
    artifacts={}
    for name in tree_names(BASE,'results'):
        filename=Path(name).name
        if not (filename.startswith(('case1_','array_','decision_study_')) or name=='results/validation.json'):continue
        data=(ROOT/name).read_bytes()
        if data!=blob(BASE,name):raise RuntimeError(f'Historical result changed since branch base: {name}')
        artifacts[name]=sha(data)
    original=blob(BASE,'notebooks/01_array.ipynb');archived=out/'01_array_base.ipynb'
    if archived.read_bytes()!=original:raise RuntimeError('Original array notebook archive changed')
    artifacts[str(archived.relative_to(ROOT))]=sha(original)
    manifest=dict(status='verified',base_commit=BASE,certificate_revision=revision,
        certificate=gate,certificate_core_paths=paths,artifacts=artifacts,
        snapshot_files=snapshots,producer_checks=producer_checks,
        interpretation='Original complex rank-three and real low-data studies; not rerun or relabelled as the new passive protocol. Archived source reconstructs the original certificate; the current beamforming gate is separate.')
    (out/'historical_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Historical certificate',revision[:8],'; preserved',len(artifacts),'artifacts and',len(snapshots),'source snapshots.')


if __name__=='__main__':main()

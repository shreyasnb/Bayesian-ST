"""Preserve the previous executed notebooks and their exact validation sources."""
from pathlib import Path
from functools import lru_cache
import ast,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
BASE='2a8697c7e7fa27b3779f3abeaed32ca8b323a9ef'
OUT=ROOT/'results/simulation_redesign/archive'

def sha(data):return hashlib.sha256(data).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
@lru_cache(None)
def blob(commit,name):return git('show',commit+':'+name)
def constant(commit,path,name):
    module=ast.parse(blob(commit,path).decode())
    return next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
def core(commit):
    paths=constant(commit,'lib/validation.py','CORE')
    tree=git('ls-tree','-r','--name-only',commit).decode().splitlines()
    paths+=sorted(p for p in tree if p.endswith('.py') and (p.startswith('benchmarks/') or p.startswith('tests/')))
    h=hashlib.sha256()
    for p in paths:h.update(p.encode());h.update(blob(commit,p))
    return h.hexdigest(),paths

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/'preservation.json'
    if target.exists():
        print('Archive already exists; verify instead of replacing it.');return
    files=git('ls-tree','-r','--name-only',BASE).decode().splitlines()
    unchanged={}
    for name in files:
        if name.startswith('results/'):
            current=(ROOT/name).read_bytes()
            if current!=blob(BASE,name):raise RuntimeError('Unrelated historical edit: '+name)
            unchanged[name]=sha(current)
    for name in ['01_array','03_contact']:
        path='notebooks/'+name+'.ipynb';data=blob(BASE,path)
        if (ROOT/path).read_bytes()!=data:raise RuntimeError('Archive notebooks before editing')
        (OUT/(name+'_2a8697c.ipynb')).write_bytes(data)
    certificates=[]
    specs=[('beamforming','results/beamforming_unified/validation/validation.json'),
           ('contact','results/contact_uncertainty/validation/validation.json')]
    commits=git('rev-list','--max-count=25',BASE).decode().splitlines()
    computed={}
    for label,path in specs:
        data=(ROOT/path).read_bytes();cert=json.loads(data);found=None
        for commit in commits:
            if commit not in computed:computed[commit]=core(commit)
            value,paths=computed[commit]
            if value==cert['core_hash']:found=(commit,paths);break
        if found is None:raise RuntimeError('Locate exact historical validation source: '+label)
        commit,paths=found;dest=OUT/(label+'_source')
        extras=list(cert.get('sources',{}))
        if label=='contact':extras+=constant(commit,'lib/contact_uncertainty_validation.py','UNCERTAINTY_SOURCES')
        for name in sorted(set(paths+extras)):
            p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(blob(commit,name))
        (OUT/(label+'_validation.json')).write_bytes(data)
        certificates.append(dict(label=label,commit=commit,core_hash=cert['core_hash'],
            core_paths=paths,source_directory=dest.name))
    root_refs={name:git('rev-parse',name).decode().strip() for name in ['main','dev','test']}
    archive_files={str(p.relative_to(OUT)):sha(p.read_bytes()) for p in sorted(OUT.rglob('*')) if p.is_file()}
    meta=dict(base_commit=BASE,branch_tips=root_refs,unchanged_results=unchanged,
        archive_files=archive_files,certificates=certificates,
        scope='Earlier notebooks/results are read-only historical evidence under exact archived producer/certificate source trees. New experiments use a new gate; no old stale certificate is bypassed.')
    target.write_text(json.dumps(meta,indent=2)+'\n')
    print('Archived',len(unchanged),'historical artifacts and',len(archive_files),'notebook/source files.')

if __name__=='__main__':main()

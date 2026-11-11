"""Update two contact plots from existing saved data, preserving other figures.

Only the directional-risk half of the original two-figure cell is executed.
The static posterior spheres retain their original source, output and files.
The second isolated kernel executes the moving-sphere prediction history only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUTPUT=Path('results/contact_plotting_addendum')
CONTACT=Path('notebooks/03_contact.ipynb')
DIAGNOSTICS=Path('notebooks/contact_prediction_diagnostics.ipynb')
TEMPLATE=Path('scripts/notebook_plot_cells_contact.py')
GENERATOR=Path('scripts/make_contact_prediction_diagnostics_notebook.py')
TARGETS={Path(folder)/(stem+suffix)
    for folder,stem in (
        ('results/notebook_exports/03_contact','posterior_anisotropy_risk'),
        ('results/main/contact','posterior_anisotropy_risk'),
        ('results/contact_prediction_diagnostics/figures','prediction_error_sphere'),
        ('results/main/contact','prediction_error_sphere'))
    for suffix in ('.pdf','.png')}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def value_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=False).encode()).hexdigest()


def source(cell):
    value=cell.get('source','')
    return ''.join(value) if isinstance(value,list) else value


def cell_by_id(notebook,identifier):
    matches=[cell for cell in notebook['cells'] if cell.get('id')==identifier]
    if len(matches)!=1:
        raise ValueError('Expected exactly one cell: '+identifier)
    return matches[0]


def protected_files(root):
    """Protect all contact evidence/figures and the frozen mean audits."""
    folders=('results/contact','results/main/contact','results/supplementary/contact',
        'results/notebook_exports/03_contact','results/contact_prediction_diagnostics',
        'results/beamforming_comparison_audit/moving/mean_validation',
        'results/beamforming_comparison_audit/stationary/mean_validation')
    paths=set()
    for folder in folders:
        paths.update(path for path in (root/folder).rglob('*') if path.is_file())
    for name in ('lib/notebook_contact_data.py','lib/contact_prediction_diagnostics.py',
        'lib/contact_moving_snapshot.py','lib/final_contact.py',
        'lib/beamforming_mean_validation_audit.py',
        'scripts/run_beamforming_mean_validation_audit.py'):
        paths.add(root/name)
    return {path.relative_to(root).as_posix():sha(path) for path in sorted(paths)
        if path.relative_to(root) not in TARGETS}


def split_risk_cell(cell):
    text=source(cell)
    end=text.index("PDF_PATH = EDITABLE_OUTPUT / 'posterior_anisotropy_risk.pdf'")
    start=text.rfind('# Edit these controls',0,end)
    if start<=0 or text[:start].count('plt.show()')!=1:
        raise ValueError('The static-sphere/risk cell no longer has the expected split')
    return text[:start],text[start:]


def patch_notebooks(root):
    from scripts.notebook_plot_cells_contact import _DIRECTIONAL_RISK_DATA
    from scripts.make_contact_prediction_diagnostics_notebook import SPHERE_HISTORY
    before={name:json.loads((root/name).read_text()) for name in (CONTACT,DIAGNOSTICS)}
    after=deepcopy(before)
    risk=cell_by_id(after[CONTACT],'45ec7869')
    first,second=split_risk_cell(risk)
    if '# Use the saved unit-query risks' not in second:
        needle='fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, layout="constrained")'
        if second.count(needle)!=1:
            raise ValueError('Expected one directional-risk plotting block')
        second=second.replace(needle,_DIRECTIONAL_RISK_DATA.strip()+'\n\n'+needle,1)
    if 'PNG_PATH = PDF_PATH.with_suffix' not in second:
        needle='    DRAFT_PATH.write_bytes(PDF_PATH.read_bytes())\nplt.show()'
        replacement='''    DRAFT_PATH.write_bytes(PDF_PATH.read_bytes())
    PNG_PATH = PDF_PATH.with_suffix('.png')
    fig.savefig(PNG_PATH, dpi=DPI, bbox_inches=None)
    DRAFT_PNG_PATH = DRAFT_PATH.with_suffix('.png')
    DRAFT_PNG_PATH.write_bytes(PNG_PATH.read_bytes())
plt.show()'''
        if second.count(needle)!=1:
            raise ValueError('Expected one directional-risk save block')
        second=second.replace(needle,replacement,1)
    risk['source']=(first+second).splitlines(keepends=True)
    note=cell_by_id(after[CONTACT],'2e836d3e')
    note['source']=source(note).replace(
        r'v_0(\cos\theta e_2+\sin\theta e_3)',
        r'v_0(\cos\theta e_1+\sin\theta e_2)').splitlines(keepends=True)
    history=cell_by_id(after[DIAGNOSTICS],'prediction-error-sphere')
    text=source(history)
    start=text.index("condition = 'sphere'")
    end=text.index('# Vector text/curves remain in the PDF;',start)
    history['source']=(text[:start]+SPHERE_HISTORY.replace('CONDITION',repr('sphere')).strip()
                       +'\n\n'+text[end:]).splitlines(keepends=True)
    note=cell_by_id(after[DIAGNOSTICS],'sphere-prediction-note')
    if 'dashed blue control' not in source(note):
        note['source']=(source(note)+r' The dashed blue control uses the saved posterior-mean '
            r'predictions with $G_t=I_3$, omitting measured sensor-frame transport '
            r'while retaining its original settings.').splitlines(keepends=True)
    from lib.contact_prediction_diagnostics import load_contact_prediction_diagnostics
    data=load_contact_prediction_diagnostics(root)
    paired=data['paired_episode_average_summary']
    identity=paired[paired.condition.eq('sphere')&paired.reference.eq('mean_identity')]
    if len(identity)!=1:
        raise ValueError('Expected one original whole-run identity-minus-mean comparison')
    row=identity.iloc[0]
    paragraph=('Across the same 48 action times, the paired identity-transport-minus-measured-transport '
        'posterior-mean error is '+f"{row['gain']:.4f} [{row['lower']:.4f}, {row['upper']:.4f}]"
        +r' $(\mathrm{mm}/\mathrm{s})^2$ (pointwise 95% interval across 256 simulations). '
        'This uses the saved whole-run episode-average records, rather than an '
        'average of the six plotted checkpoints.')
    results=cell_by_id(after[DIAGNOSTICS],'contact-prediction-results')
    old=source(results).split('\n\nAcross the same 48 action times,')[0]
    results['source']=(old+'\n\n'+paragraph).splitlines(keepends=True)
    checks=dict(identity_episode_average_gain=float(row['gain']),
        identity_episode_average_lower=float(row['lower']),
        identity_episode_average_upper=float(row['upper']),simulations=int(row['simulations']),
        averaging_source='Existing paired_episode_average_summary at endpoint episode_average, '
            't192; all 48 action times, not the six plotted checkpoints.',
        first_static_sphere_source_sha256=hashlib.sha256(first.encode()).hexdigest(),
        first_static_sphere_output_sha256=value_sha(risk['outputs'][0]))
    return before,after,second,checks,data['input_hashes']


def execute_plot(root,setup,plot,directory,name,timeout):
    import nbformat
    from nbclient import NotebookClient
    os.environ.setdefault('MPLCONFIGDIR','/tmp/bayesian-st-matplotlib')
    os.environ.setdefault('JUPYTER_RUNTIME_DIR','/tmp/bayesian-st-jupyter')
    os.environ.setdefault('IPYTHONDIR','/tmp/bayesian-st-ipython')
    notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(setup),
        nbformat.v4.new_code_cell(plot)],metadata={
            'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    NotebookClient(notebook,timeout=timeout,kernel_name='python3',
        resources={'metadata':{'path':str(root)}}).execute()
    nbformat.write(notebook,Path(directory)/(name+'.ipynb'))
    outputs=notebook.cells[1].outputs
    displays=[item for item in outputs if item.output_type in ('display_data','execute_result')]
    if len(displays)!=1 or 'image/png' not in displays[0].get('data',{}):
        raise AssertionError('Expected one rendered plot: '+name)
    return json.loads(json.dumps(outputs))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--timeout',type=int,default=900)
    args=parser.parse_args();root=args.root.resolve()
    baseline=protected_files(root)
    old_hashes={p.as_posix():sha(root/p) for p in (CONTACT,DIAGNOSTICS)}
    old_hashes.update({p.as_posix():sha(root/p) for p in TARGETS if (root/p).exists()})
    before,after,risk_source,checks,input_hashes=patch_notebooks(root)
    import numpy as np
    path=root/'results/contact/data/paper/performance/anisotropy.npz'
    with np.load(path,allow_pickle=False) as saved:
        selected=saved['query_family']==0
        theta=saved['query_angles'][selected]
        risk=saved['risks_full'][selected];iso=saved['risks_isotropic'][selected]
        endpoints=[]
        for angle,expected in ((0.,(.019462209657903684,.08842183901123923)),
                               (np.pi/2,(.14346703590668408,.09824824499508511))):
            index=int(np.argmin(abs(theta-angle)))
            np.testing.assert_allclose([risk[index],iso[index]],expected,rtol=0,atol=1e-14)
            endpoints.append(dict(theta=float(theta[index]),risk_full=float(risk[index]),
                                  risk_isotropic=float(iso[index])))
    checks.update(query_family=0,query_definition='v(theta)=v0*(cos(theta)*e1+sin(theta)*e2)',
        risk_definition='Saved unit-input risks directly equal r_t(v(theta))/v0^2; no rescaling.',
        directional_risk_endpoints=endpoints,new_simulations=0,new_parameter_selections=0,
        new_posterior_fits=0)
    if not args.execute:
        print(json.dumps(dict(status='ready_to_execute',checks=checks,
            notebooks=[str(CONTACT),str(DIAGNOSTICS)],plots=2),indent=2))
        return
    with TemporaryDirectory(prefix='contact-plot-addendum-') as directory:
        risk_outputs=execute_plot(root,source(before[CONTACT]['cells'][1]),risk_source,
                                  directory,'directional_risk',args.timeout)
        risk_cell=cell_by_id(after[CONTACT],'45ec7869')
        previous=cell_by_id(before[CONTACT],'45ec7869')
        if len(previous['outputs'])!=2:
            raise ValueError('Expected exactly two original figure outputs')
        risk_cell['outputs']=deepcopy(previous['outputs'][:1])+risk_outputs
        history=cell_by_id(after[DIAGNOSTICS],'prediction-error-sphere')
        setup=cell_by_id(before[DIAGNOSTICS],'contact-diagnostics-setup')
        history['outputs']=execute_plot(root,source(setup),source(history),directory,
                                        'sphere_prediction',args.timeout)
    allowed={CONTACT:{'45ec7869','2e836d3e'},DIAGNOSTICS:{'prediction-error-sphere',
        'sphere-prediction-note','contact-prediction-results'}}
    for name in (CONTACT,DIAGNOSTICS):
        if len(before[name]['cells'])!=len(after[name]['cells']):
            raise AssertionError('Notebook cells were added or removed')
        if before[name]['metadata']!=after[name]['metadata']:
            raise AssertionError('Unrelated notebook metadata changed')
        for old,new in zip(before[name]['cells'],after[name]['cells']):
            if old.get('id') not in allowed[name] and old!=new:
                raise AssertionError('Unrelated notebook cell changed: '+str(old.get('id')))
    first,_=split_risk_cell(cell_by_id(after[CONTACT],'45ec7869'))
    if hashlib.sha256(first.encode()).hexdigest()!=checks['first_static_sphere_source_sha256']:
        raise AssertionError('Static-sphere source changed')
    if value_sha(cell_by_id(after[CONTACT],'45ec7869')['outputs'][0])!=checks['first_static_sphere_output_sha256']:
        raise AssertionError('Static-sphere output changed')
    changed=[name for name,digest in baseline.items() if not (root/name).is_file() or sha(root/name)!=digest]
    if changed:
        raise AssertionError('Protected evidence/figure changed: '+', '.join(changed))
    for local,draft in (
        ('results/notebook_exports/03_contact/posterior_anisotropy_risk',
         'results/main/contact/posterior_anisotropy_risk'),
        ('results/contact_prediction_diagnostics/figures/prediction_error_sphere',
         'results/main/contact/prediction_error_sphere')):
        for suffix in ('.pdf','.png'):
            if sha(root/(local+suffix))!=sha(root/(draft+suffix)):
                raise AssertionError('Local/manuscript export mismatch')
    for name in (CONTACT,DIAGNOSTICS):
        (root/name).write_text(json.dumps(after[name],indent=1,ensure_ascii=False)+'\n')
    out=root/OUTPUT;out.mkdir(parents=True,exist_ok=True)
    modified={name:dict(before=digest,after=sha(root/name)) for name,digest in old_hashes.items()}
    for relative in TARGETS:
        name=relative.as_posix()
        if name not in modified:
            modified[name]=dict(before=None,after=sha(root/relative))
    report=dict(status='passed',checks=checks,modified_files=modified,
        protected_files_checked=len(baseline),protected_file_hashes=baseline,
        untouched_cell_ids={str(name):[c.get('id') for c in before[name]['cells']
            if c.get('id') not in allowed[name]] for name in (CONTACT,DIAGNOSTICS)},
        scope='Only saved query family0 and the existing identity-transport prediction '
            'curve were exposed. Static spheres, other notebook cells, numerical '
            'procedures, cohorts, settings and old manifests are unchanged.')
    report_path=out/'report.json';report_path.write_text(json.dumps(report,indent=2)+'\n')
    sources={str(p):sha(root/p) for p in (Path(__file__).relative_to(root),TEMPLATE,GENERATOR)}
    manifest=dict(status='completed',producer=Path(__file__).relative_to(root).as_posix(),
        source_hashes=sources,input_hashes=dict(input_hashes,**{str(path.relative_to(root)):sha(path)}),
        outputs={str(p):sha(root/p) for p in sorted(TARGETS|{CONTACT,DIAGNOSTICS})},
        report_path=str(report_path.relative_to(root)),report_sha256=sha(report_path),
        plotting=dict(dpi=300,figure_size_in=[15,5],text_usetex=True),checks=checks,
        provenance='Independent presentation addendum. Existing data/figure manifests '
            'are intentionally unchanged; this manifest records only the authorized plot update.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status='completed',checks=checks,modified_files=modified,
        protected_files_checked=len(baseline),manifest=str(OUTPUT/'manifest.json')),indent=2))


if __name__=='__main__':
    main()

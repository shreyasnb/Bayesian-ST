"""Render only the three 2D contact diagnostic figures with full plot borders."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.update_contact_plotting_addendum import sha,value_sha,source,cell_by_id

NOTEBOOK=Path('notebooks/contact_prediction_diagnostics.ipynb')
OUTPUT=Path('results/contact_plotting_addendum')
GENERATOR=Path('scripts/make_contact_prediction_diagnostics_notebook.py')
PLOTS=(('prediction-error-stationary','prediction_error_stationary','main'),
       ('prediction-error-sphere','prediction_error_sphere','main'),
       ('prediction-error-references','prediction_error_references','supplementary'))
BORDER_CODE="""BORDER_WIDTH = .9
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(BORDER_WIDTH)
ax.tick_params(axis='both', which='both', direction='in',
               top=True, right=True, bottom=True, left=True,
               labeltop=False, labelright=False, labelbottom=True, labelleft=True,
               width=BORDER_WIDTH)
ax.xaxis.set_label_position('bottom')
ax.yaxis.set_label_position('left')"""


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--timeout',type=int,default=900)
    args=parser.parse_args();root=args.root.resolve()
    targets={Path(folder)/(stem+suffix) for _,stem,section in PLOTS
        for folder in ('results/contact_prediction_diagnostics/figures',
                       'results/'+section+'/contact') for suffix in ('.pdf','.png')}
    protected=set()
    for folder in ('results/contact/data','results/contact_prediction_diagnostics',
        'results/beamforming_comparison_audit/moving/mean_validation',
        'results/beamforming_comparison_audit/stationary/mean_validation'):
        protected.update(path for path in (root/folder).rglob('*') if path.is_file())
    for name in ('lib/contact_prediction_diagnostics.py','lib/contact_moving_snapshot.py',
        'lib/notebook_contact_data.py','lib/final_contact.py',
        'lib/beamforming_mean_validation_audit.py',
        'scripts/run_beamforming_mean_validation_audit.py'):
        protected.add(root/name)
    for folder in ('results/contact_prediction_diagnostics/figures','results/supplementary/contact'):
        for suffix in ('.pdf','.png'):
            protected.add(root/folder/('moving_sphere_posterior_spheres'+suffix))
    baseline={str(p.relative_to(root)):sha(p) for p in sorted(protected)
              if p.relative_to(root) not in targets}
    before=json.loads((root/NOTEBOOK).read_text());after=deepcopy(before)
    old_hashes={str(NOTEBOOK):sha(root/NOTEBOOK),
                **{str(p):sha(root/p) for p in sorted(targets)}}
    for identifier,_,_ in PLOTS:
        cell=cell_by_id(after,identifier);text=source(cell)
        old="ax.spines[['top','right']].set_visible(False)"
        if BORDER_CODE not in text:
            if text.count(old)!=1:
                raise ValueError('Expected one 2D spine instruction: '+identifier)
            text=text.replace(old,BORDER_CODE)
        cell['source']=text.splitlines(keepends=True)
        if 'FIGSIZE = (15, 5)' not in text or 'DPI = 300' not in text:
            raise AssertionError('Figure controls changed')
        if "spines[['top','right']].set_visible(False)" in text:
            raise AssertionError('A 2D border is still hidden')
    import nbformat
    from nbclient import NotebookClient
    setup=cell_by_id(before,'contact-diagnostics-setup')
    cells=[nbformat.v4.new_code_cell(source(setup))]
    for identifier,stem,_ in PLOTS:
        cells += [nbformat.v4.new_markdown_cell(stem),
                  nbformat.v4.new_code_cell(source(cell_by_id(after,identifier)))]
    # Seven cells: original setup, then three markdown/plot pairs; no other code.
    mini=nbformat.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    os.environ.setdefault('MPLCONFIGDIR','/tmp/bayesian-st-matplotlib')
    os.environ.setdefault('JUPYTER_RUNTIME_DIR','/tmp/bayesian-st-jupyter')
    os.environ.setdefault('IPYTHONDIR','/tmp/bayesian-st-ipython')
    with TemporaryDirectory(prefix='contact-diagnostics-borders-') as directory:
        NotebookClient(mini,timeout=args.timeout,kernel_name='python3',
            resources={'metadata':{'path':str(root)}}).execute()
        nbformat.write(mini,Path(directory)/'three_plots.ipynb')
    for index,(identifier,_,_) in enumerate(PLOTS):
        outputs=json.loads(json.dumps(mini.cells[2+2*index].outputs))
        if len([x for x in outputs if 'image/png' in x.get('data',{})])!=1:
            raise AssertionError('Expected one figure output: '+identifier)
        cell_by_id(after,identifier)['outputs']=outputs
    allowed={x[0] for x in PLOTS}
    if before['metadata']!=after['metadata'] or len(before['cells'])!=len(after['cells']):
        raise AssertionError('Unrelated notebook structure/metadata changed')
    for old,new in zip(before['cells'],after['cells']):
        if old.get('id') not in allowed and old!=new:
            raise AssertionError('Unrelated notebook cell changed: '+str(old.get('id')))
    unchanged=[name for name,digest in baseline.items() if not (root/name).exists() or sha(root/name)!=digest]
    if unchanged:
        raise AssertionError('Protected contact/mean-audit evidence changed: '+', '.join(unchanged))
    for _,stem,section in PLOTS:
        for suffix in ('.pdf','.png'):
            local=root/'results/contact_prediction_diagnostics/figures'/(stem+suffix)
            draft=root/'results'/section/'contact'/(stem+suffix)
            if sha(local)!=sha(draft):
                raise AssertionError('Local/manuscript export mismatch: '+stem)
    (root/NOTEBOOK).write_text(json.dumps(after,indent=1,ensure_ascii=False)+'\n')
    out=root/OUTPUT;out.mkdir(parents=True,exist_ok=True)
    sphere=cell_by_id(before,'moving-sphere-posterior-spheres')
    report=dict(status='passed',notebook=str(NOTEBOOK),plot_cell_ids=sorted(allowed),
        border=dict(width_points=.9,four_spines=True,ticks='inward on all four sides',
                    tick_labels='bottom and left only'),
        unchanged_controls=dict(dpi=300,figure_size_in=[15,5],text_usetex=True),
        moving_density_cell_sha256=value_sha(sphere),
        changed_files={name:dict(before=digest,after=sha(root/name))
                       for name,digest in old_hashes.items()},
        protected_files_checked=len(baseline),protected_file_hashes=baseline,
        unchanged_cell_ids=[c.get('id') for c in before['cells'] if c.get('id') not in allowed],
        new_simulations=0,new_parameter_selections=0,new_posterior_fits=0,
        scope='Only plot borders and tick directions changed. Original curves, '
            'confidence intervals, identity control, labels, canvas sizes and fonts '
            'are retained. Moving 3D density source/output/PDF/PNG and all data remain exact.')
    path=out/'diagnostics_borders.json';path.write_text(json.dumps(report,indent=2)+'\n')
    producer=Path(__file__).resolve().relative_to(root)
    manifest=dict(status='completed',producer=str(producer),
        source_hashes={str(p):sha(root/p) for p in (producer,GENERATOR,
            Path('scripts/update_contact_plotting_addendum.py'))},
        outputs={str(p):sha(root/p) for p in sorted(targets|{NOTEBOOK})},
        report=str(path.relative_to(root)),report_sha256=sha(path),
        scope=report['scope'])
    (out/'diagnostics_borders_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status='completed',plot_cell_ids=sorted(allowed),
        protected_files_checked=len(baseline),changed_files=report['changed_files']),indent=2))


if __name__=='__main__':
    main()

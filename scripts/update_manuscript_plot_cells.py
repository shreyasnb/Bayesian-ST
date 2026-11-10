"""Refresh application plotting cells from the revised manuscript templates.

Keeps the notebook cell order/IDs and numerical check cells. This explicit
presentation-update command resets the two application plotting templates;
normal notebook execution never rewrites source or changes numerical evidence.
"""
from __future__ import annotations
import argparse
import hashlib
from importlib import import_module
import json
import re
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.make_final_notebooks import editable_setup, MAIN_PLOT_NOTES, paper_findings
from scripts.append_supplementary_notebooks import FAMILIES, NOTEBOOKS, SETUP


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def update(example):
    path=ROOT/'notebooks'/NOTEBOOKS[example]
    nb=json.loads(path.read_text())
    start=nb['metadata']['supplementary_appendix']['original_cell_count']
    main=import_module('scripts.notebook_plot_cells_'+example)
    supplementary={(r['stem'],r['page']):r for family in FAMILIES
        for r in import_module('scripts.notebook_supp_'+family).build_cells(example)}
    notes=paper_findings(ROOT,example)
    updated=0
    for index,cell in enumerate(nb['cells']):
        info=cell.get('metadata',{})
        text=''.join(cell['source'])
        if info.get('editable_plotting_role')=='plot':
            key=(info['figure_id'],int(info.get('source_page',1)))
            if index<start:
                source=main.get_plot_cell(*key)
            else:
                source=supplementary[key]['source']
                previous=nb['cells'][index-1]
                record=supplementary[key]
                previous['source']=('### '+record['title']+'\n\n'+record['note']+'\n').splitlines(True)
            cell['source']=source.splitlines(True)
            cell['execution_count']=None;cell['outputs']=[]
            updated+=1
        elif index==1 and cell['cell_type']=='code':
            cell['source']=editable_setup(example).splitlines(True)
        elif info.get('supplementary_setup'):
            source=SETUP.replace('__EXAMPLE__',repr(example)).replace('__STEM__',repr(Path(NOTEBOOKS[example]).stem))
            cell['source']=source.splitlines(True)
        elif cell['cell_type']=='markdown' and index<start:
            text=text.replace('d_{\\mathrm{ch}}','d_c').replace('d_{ch}','d_c')
            text=text.replace('independent episodes','Monte Carlo simulations').replace('repeated-episode','Monte Carlo')
            text=text.replace('MVDR with covariance shrinkage','Regularized sample covariance')
            text=text.replace('Loaded EW covariance','Regularized sample covariance')
            text=text.replace('$p_Z$','$p_z$').replace('p_Z)','p_z)')
            if text.startswith('## 3. Task decisions and outcomes'):
                if MAIN_PLOT_NOTES[example+'_task'].strip() not in text:
                    text+='\n\n'+MAIN_PLOT_NOTES[example+'_task']
            if text.startswith('## 4. Which uncertainty') and example=='contact':
                # Replace the old unit-input description by the physical normalized definition.
                split=' In the contact risk panel,'
                if split in text:text=text.split(split)[0]
                text+='\n\n'+MAIN_PLOT_NOTES['contact_risk']
            if text.startswith('## 5. Learning variances') and example=='beamforming':
                if 'Here $n=32$, $T=256$' in text:
                    text=text.split('Here $n=32$, $T=256$')[0]
                text+='\n\n'+MAIN_PLOT_NOTES['beamforming_noise']
            if text.startswith('The table reports reference loss minus posterior-mean loss'):
                text=notes['task']
            if example=='contact' and text.startswith('Panel (a) uses stationary concentrated inputs.'):
                text=MAIN_PLOT_NOTES['contact_task']
            if text.startswith('Define the matched isotropic mean by:'):
                text=r'''Define the matched isotropic mean by
\[
\zeta_t=k-\operatorname{tr}(\overline P_t^+P_{\widehat{\mathcal X}_t}),\qquad
\overline P_t^{\mathrm{iso}}=
\left(1-\frac{\zeta_t}{k}\right)P_{\widehat{\mathcal X}_t}
+\frac{\zeta_t}{n-k}(I_n-P_{\widehat{\mathcal X}_t}).
\]
It preserves the modal subspace and posterior expected squared chordal distance,
while assigning equal eigenvalues within the subspace and its complement.
'''
            text=text.replace('episode-average', 'time-averaged').replace('within-episode', 'within-simulation')
            text=re.sub(r'\bepisodes\b', 'Monte Carlo simulations', text)
            text=re.sub(r'\bepisode\b', 'simulation', text)
            cell['source']=text.splitlines(True)
    for name in nb['metadata']['editable_plotting']['loader_sources']:
        nb['metadata']['editable_plotting']['loader_sources'][name]=sha(ROOT/name)
    appendix=nb['metadata']['supplementary_appendix']
    appendix['generation_sources']={f'scripts/notebook_supp_{family}.py':sha(ROOT/f'scripts/notebook_supp_{family}.py') for family in FAMILIES}
    appendix['generation_sources']['scripts/notebook_publication_style.py']=sha(ROOT/'scripts/notebook_publication_style.py')
    appendix['presentation_revision']='Revised manuscript notation, exact saved array power, and fonts for six-inch insertion. Numerical records and numerical-check cells unchanged.'
    meta=nb['metadata'].get('final_study',{})
    meta['producer_sha256']=sha(ROOT/'scripts/make_final_notebooks.py')
    manifest=ROOT/'results'/example/'figure_manifest.json'
    meta['figure_manifest_sha256']=sha(manifest)
    for name in meta.get('reference_previews',{}):meta['reference_previews'][name]=sha(ROOT/name)
    appendix['source_figure_manifest_sha256']=sha(manifest)
    path.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n')
    print(json.dumps(dict(notebook=path.name,updated_plot_cells=updated,cell_count=len(nb['cells']))))



def refresh_metadata(example):
    """Refresh provenance only after verifying source templates still agree."""
    path=ROOT/'notebooks'/NOTEBOOKS[example]
    nb=json.loads(path.read_text())
    start=nb['metadata']['supplementary_appendix']['original_cell_count']
    main=import_module('scripts.notebook_plot_cells_'+example)
    records={(r['stem'],r['page']):r for family in FAMILIES
             for r in import_module('scripts.notebook_supp_'+family).build_cells(example)}
    for index,cell in enumerate(nb['cells']):
        info=cell.get('metadata',{})
        if info.get('editable_plotting_role')!='plot':continue
        key=(info['figure_id'],int(info.get('source_page',1)))
        expected=main.get_plot_cell(*key) if index<start else records[key]['source']
        assert ''.join(cell['source'])==expected, 'Refresh the changed plotting cell before rebinding provenance: '+str(key)
        assert cell['execution_count'] is not None and not any(o['output_type']=='error' for o in cell['outputs'])
        if index>=start:
            record=records[key]
            nb['cells'][index-1]['source']=('### '+record['title']+'\n\n'+record['note']+'\n').splitlines(True)
    meta=nb['metadata']['final_study']
    meta['producer_sha256']=sha(ROOT/'scripts/make_final_notebooks.py')
    meta['figure_manifest_sha256']=sha(ROOT/'results'/example/'figure_manifest.json')
    for name in meta.get('reference_previews',{}):meta['reference_previews'][name]=sha(ROOT/name)
    appendix=nb['metadata']['supplementary_appendix']
    appendix['source_figure_manifest_sha256']=meta['figure_manifest_sha256']
    for name in appendix['generation_sources']:appendix['generation_sources'][name]=sha(ROOT/name)
    for name,digest in nb['metadata']['editable_plotting']['loader_sources'].items():
        assert sha(ROOT/name)==digest, 'An executed notebook reader changed: '+name
    path.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n')
    print('Verified executed templates and refreshed presentation provenance:',path.name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example',choices=NOTEBOOKS,action='append')
    parser.add_argument('--metadata-only', action='store_true', help='Verify executed figure sources, then refresh presentation provenance and notes only.')
    args=parser.parse_args()
    for example in args.example or NOTEBOOKS:
        (refresh_metadata if args.metadata_only else update)(example)

if __name__=='__main__':main()

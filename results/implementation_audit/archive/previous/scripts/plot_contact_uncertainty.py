"""Render the separate contact-uncertainty studies after source/table checks."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from lib.plotting import plt
from lib.contact_uncertainty_validation import require_uncertainty_gate
from lib.contact_uncertainty_visuals import (coverage_score,paired_effects_cost,
    directional_calibration,validation_scores,shape_concentration,physical_stress)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_study(root,directory,gate):
    root=Path(root);directory=Path(directory);metadata=json.loads((directory/'config.json').read_text())
    if metadata.get('development_only'):raise RuntimeError('Development smoke data are not the comparison')
    captured=metadata.get('gate')
    if not captured or any(captured[key]!=gate[key] for key in ['core_hash','uncertainty_hash']):
        raise RuntimeError(f'Study captured a different numerical gate: {directory}')
    for name,digest in metadata['sources'].items():
        if sha(root/name)!=digest:raise RuntimeError(f'Stale study source: {name}')
    frames=dict(config=metadata['config'],metadata=metadata)
    for name,digest in metadata['tables'].items():
        path=directory/name
        if sha(path)!=digest:raise RuntimeError(f'Changed study table: {path}')
        if path.suffix=='.csv':
            try:frames[path.stem]=pd.read_csv(path)
            except pd.errors.EmptyDataError:frames[path.stem]=pd.DataFrame()
    return frames


def load_comparison(root=ROOT,directory=None):
    root=Path(root).resolve();directory=Path(directory or root/'results/contact_uncertainty').resolve()
    gate=require_uncertainty_gate(root)
    physical=load_study(root,directory/'physical',gate)
    matched=load_study(root,directory/'matched',gate)
    reference=json.loads((directory/'reference/checks.json').read_text())
    for name,digest in reference.get('sources',{}).items():
        if sha(root/name)!=digest:raise RuntimeError(f'Stale reference development source: {name}')
    for name,digest in reference.get('tables',{}).items():
        if sha(directory/'reference'/name)!=digest:raise RuntimeError(f'Changed reference check table: {name}')
    return physical,matched,reference,gate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=ROOT/'results/contact_uncertainty')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();directory=args.results_dir.resolve()
    physical,matched,reference,gate=load_comparison(ROOT,directory)
    out=(args.output_dir or directory/'figures').resolve();out.mkdir(parents=True,exist_ok=True)
    products=[]
    def render(function,name,*items,**kwargs):
        fig=function(*items,path=out/name,**kwargs);plt.close(fig)
        products.extend([out/f'{name}.pdf',out/f'{name}.png'])
        print('Rendered',name,flush=True)
    render(coverage_score,'coverage_score',physical,matched)
    render(paired_effects_cost,'effects_cost',physical)
    render(directional_calibration,'directional_calibration',physical)
    render(validation_scores,'validation_scores',physical)
    render(shape_concentration,'shape_concentration',physical)
    render(physical_stress,'physical_stress',physical)
    render(coverage_score,'plane_coverage',physical,condition='plane')
    render(coverage_score,'weak_coverage',physical,condition='weak')
    record=dict(core_hash=gate['core_hash'],uncertainty_hash=gate['uncertainty_hash'],
        sources={name:sha(ROOT/name) for name in ['lib/contact_uncertainty_visuals.py','scripts/plot_contact_uncertainty.py','lib/plotting.py']},
        input_metadata={str((directory/name).relative_to(ROOT)):sha(directory/name) for name in
                        ['physical/config.json','matched/config.json','reference/checks.json']},
        figures={path.name:sha(path) for path in products},dpi=200,text_usetex=True,
        main_figures=['coverage_score','effects_cost','directional_calibration'],
        reference_status=reference.get('status'),
        physical_numerical_checks=physical['metadata'].get('checks',{}),
        matched_numerical_checks=matched['metadata'].get('checks',{}),
        interpretation='Process assumptions, same-parameter shape, approximation limitations and numerical resolution have separate labels; no numerical failure is converted into a positive claim.')
    (out/'manifest.json').write_text(json.dumps(record,indent=2))


if __name__=='__main__':main()

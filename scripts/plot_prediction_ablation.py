"""Render the isolated checkpoint studies after the extended numerical gate.

Original notebooks and results are never changed. Every table is verified
against the metadata saved by its numerical producer before rendering.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from lib.validation import require_gate
from lib.plotting import plt, DPI
from lib.prediction_visuals import (motion_overview,application_gains,
    process_coverage_width,motion_coverage_width,memory_tuning,motion_stress,
    application_tuning,application_cost,process_prediction_fans)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_study(directory,metadata_name='config.json'):
    directory=Path(directory);meta=json.loads((directory/metadata_name).read_text())
    for group in ['sources','postprocessing_sources']:
        for relative,digest in meta.get(group,{}).items():
            if sha(ROOT/relative)!=digest:raise RuntimeError(f'Stale numerical source: {relative}')
    tables={}
    for name,digest in meta['tables'].items():
        target=directory/name
        if sha(target)!=digest:raise RuntimeError(f'Table checksum changed: {target}')
        if target.suffix=='.csv':
            try:tables[target.stem]=pd.read_csv(target)
            except pd.errors.EmptyDataError:tables[target.stem]=pd.DataFrame()
    return tables,meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=ROOT/'results/dev_prediction')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();directory=args.results_dir.resolve()
    gate=require_gate(ROOT,gate_path=directory/'validation.json')
    output=(args.output_dir or directory/'figures').resolve();output.mkdir(parents=True,exist_ok=True)
    motion,motion_meta=load_study(directory/'motion')
    applications,application_meta=load_study(directory/'applications')
    process,process_meta=load_study(directory/'process','status.json')
    binding_path=directory/'gate_binding.json'
    binding=json.loads(binding_path.read_text())
    if binding['final_core_hash']!=gate['core_hash']:raise RuntimeError('Final gate binding is stale')
    for name,meta in [('motion',motion_meta),('applications',application_meta),('process',process_meta)]:
        if name in binding['captured_run_core_hashes']:
            assert binding['captured_run_core_hashes'][name]==meta.get('core_hash'),name
        assert binding['verified_sources'][name]=={**meta['sources'],**meta.get('postprocessing_sources',{})},name
        assert binding['verified_tables'][name]==meta['tables'],name
    if len(motion['posterior']):
        paired_meta=json.loads((directory/'motion/paired_noise_evaluation.json').read_text())
        assert paired_meta['status']=='checked'
        assert paired_meta['script_sha256']==sha(ROOT/'scripts/check_motion_paired_noise.py')
        assert paired_meta['input_sha256']==sha(directory/'motion/posterior.csv')
        assert paired_meta['original_metadata_sha256']==sha(directory/'motion/config_before_paired_noise.json')
        assert paired_meta['original_producer_source_hash']==motion_meta['source_hash']
        for name,digest in paired_meta['tables'].items():
            assert sha(directory/'motion'/name)==digest,name
        motion['posterior']=motion['posterior_paired_noise']
        motion['coverage']=motion['coverage_paired_noise']
    products=[]
    def render(function,name,*items,**kwargs):
        fig=function(*items,path=output/name,**kwargs);plt.close(fig)
        products.extend([output/f'{name}.pdf',output/f'{name}.png'])
        print('Rendered',name,flush=True)
    render(motion_overview,'motion',motion)
    render(application_gains,'application_gains',applications['paired'],tuning=applications['tuning'])
    render(process_coverage_width,'process_coverage_width',process['coverage_summary'])
    render(memory_tuning,'memory_tuning',motion['tuning'],motion_meta['selected'])
    render(motion_stress,'motion_stress',motion['summary'])
    if len(motion['coverage']):render(motion_coverage_width,'motion_coverage_width',motion['coverage'])
    render(application_gains,'application_stress',applications['paired'],all_scenarios=True,tuning=applications['tuning'])
    for application in ['array','field']:
        render(application_tuning,application+'_tuning',applications['tuning'],application)
    render(application_cost,'application_cost',applications['metrics'])
    render(process_prediction_fans,'process_predictions',process['paths'],process['coverage_trials'])
    for law,name in [('Slow deterministic','process_slow'),('Fast deterministic','process_fast'),
                     ('Acceleration and turn','process_turn'),('Observation contamination','process_contamination')]:
        render(process_coverage_width,name,process['coverage_summary'],law=law,include_controls=True)
    record=dict(core_hash=gate['core_hash'],gate_binding_sha256=sha(binding_path),
        sources={p:sha(ROOT/p) for p in ['lib/prediction_visuals.py','scripts/plot_prediction_ablation.py','lib/plotting.py']},
        input_metadata={str((directory/study/name).relative_to(ROOT)):sha(directory/study/name) for study,name in
                        [('motion','config.json'),('applications','config.json'),('process','status.json')]},
        figures={p.name:sha(p) for p in products},dpi=DPI,text_usetex=True,
        motion_query_tables='posterior_paired_noise.csv and coverage_paired_noise.csv; raw inherited query evaluations preserved',
        main_figures=['motion','application_gains','process_coverage_width'],
        interpretation='Replicated-data intervals, maintained-posterior uncertainty, and numerical reference errors have separate labels.')
    (output/'manifest.json').write_text(json.dumps(record,indent=2))


if __name__=='__main__':
    main()

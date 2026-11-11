"""Export verified moving-interference summaries without rerunning a simulation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lib.beamforming_moving_evidence import load_moving_evidence, sha, OUTPUT


def export(root=ROOT):
    root = Path(root).resolve()
    evidence = load_moving_evidence(root)
    base = root/OUTPUT
    data,tables = base/'data',base/'tables'
    data.mkdir(parents=True,exist_ok=True);tables.mkdir(parents=True,exist_ok=True)
    def json_file(path,value):
        path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    for key,filename in [('config','config.json'),('checks','checks.json'),
                         ('provenance','provenance.json'),('selected_settings','selected_settings.json')]:
        json_file(data/filename,evidence[key])
    evidence['checkpoint_losses'].to_csv(data/'checkpoint_losses.csv.gz',index=False,
        compression={'method':'gzip','mtime':0})
    for key in ('checkpoint_summary','paired_checkpoint_summary',
                'paired_checkpoint_differences','simulation_time_averages',
                'time_average_summary','paired_time_average_summary','paired_time_average_differences'):
        compress = key=='paired_checkpoint_differences'
        target = tables/(key+('.csv.gz' if compress else '.csv'))
        evidence[key].to_csv(target,index=False,
            compression={'method':'gzip','mtime':0} if compress else None)
    artifacts = {path.relative_to(root).as_posix():sha(path)
                 for folder in (data,tables) for path in sorted(folder.iterdir())
                 if path.is_file() and path.name!='manifest.json'}
    manifest = dict(schema=1,status='completed',example='beamforming_moving',
        component='original_moving_interference',config=evidence['config'],
        producer='scripts/summarize_moving_beamforming.py',
        sources={name:sha(root/name) for name in (
            'scripts/summarize_moving_beamforming.py','lib/beamforming_moving_evidence.py')},
        inputs=evidence['input_hashes'],original_sources=evidence['source_hashes'],
        artifacts=artifacts,checks=evidence['checks'],
        scope='Derived summaries of immutable original observations; no new simulation, posterior inference, fitting, parameter selection or scientific-source edits.')
    json_file(data/'manifest.json',manifest)
    return dict(status='completed',manifest=(data/'manifest.json').relative_to(root).as_posix(),
        artifacts=len(artifacts),raw_simulation_rows=len(evidence['checkpoint_losses']),
        checkpoint_summaries=len(evidence['checkpoint_summary']),
        paired_checkpoint_summaries=len(evidence['paired_checkpoint_summary']),
        paired_checkpoint_differences=len(evidence['paired_checkpoint_differences']),
        time_average_summaries=len(evidence['time_average_summary']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    print(json.dumps(export(args.root),indent=2))


if __name__=='__main__':
    main()

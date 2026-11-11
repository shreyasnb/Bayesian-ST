"""Export verified contact diagnostics without altering original experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lib.contact_prediction_diagnostics import load_contact_prediction_diagnostics,sha,OUTPUT


def export(root=ROOT):
    root=Path(root).resolve();evidence=load_contact_prediction_diagnostics(root)
    data=root/OUTPUT/'data';tables=root/OUTPUT/'tables'
    data.mkdir(parents=True,exist_ok=True);tables.mkdir(parents=True,exist_ok=True)
    def json_file(path,value):
        path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    for key in ('config','checks','provenance','selected_settings'):
        json_file(data/(key+'.json'),evidence[key])
    evidence['raw_stats'].to_csv(data/'raw_stats.csv.gz',index=False,
        compression={'method':'gzip','mtime':0})
    for key in ('checkpoint_summary','episode_average_summary','paired_checkpoint_summary',
                'paired_episode_average_summary','paired_checkpoint_differences','paired_episode_average_differences'):
        compressed=key.endswith('_differences')
        evidence[key].to_csv(tables/(key+('.csv.gz' if compressed else '.csv')),index=False,
            compression={'method':'gzip','mtime':0} if compressed else None)
    artifacts={path.relative_to(root).as_posix():sha(path)
        for folder in (data,tables) for path in sorted(folder.iterdir())
        if path.is_file() and path.name!='manifest.json'}
    manifest=dict(schema=1,status='completed',example='contact_prediction_diagnostics',
        config=evidence['config'],producer='scripts/export_contact_prediction_diagnostics.py',
        sources={name:sha(root/name) for name in ('lib/contact_prediction_diagnostics.py',
            'scripts/export_contact_prediction_diagnostics.py')},
        inputs=evidence['input_hashes'],original_sources=evidence['source_hashes'],
        artifacts=artifacts,checks=evidence['checks'],
        scope='Read-only summaries and algebraic checks of original saved contact results; no new simulations, fitting, selection or scientific-source edits.')
    json_file(data/'manifest.json',manifest)
    return dict(status='completed',manifest=(data/'manifest.json').relative_to(root).as_posix(),
        artifacts=len(artifacts),raw_rows=len(evidence['raw_stats']),
        checkpoint_summaries=len(evidence['checkpoint_summary']),
        episode_average_summaries=len(evidence['episode_average_summary']),
        paired_checkpoint_summaries=len(evidence['paired_checkpoint_summary']),
        paired_episode_average_summaries=len(evidence['paired_episode_average_summary']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    print(json.dumps(export(args.root),indent=2))


if __name__=='__main__':main()

"""Export existing-data beamforming comparisons, preserving original outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lib.beamforming_comparison_summaries import load_existing_beamforming_comparisons,sha,OUTPUT


def export(root=ROOT):
    root=Path(root).resolve();evidence=load_existing_beamforming_comparisons(root);results={}
    for case,data in evidence.items():
        folder=root/OUTPUT/case/'existing';folder.mkdir(parents=True,exist_ok=True)
        def json_file(filename,value):
            (folder/filename).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
        for key in ('config','provenance','checks','selected_settings'):
            json_file(key+'.json',data[key])
        names=['checkpoint_summary','paired_checkpoint_summary','paired_checkpoint_differences',
               'primary_summary','paired_primary_summary']
        if case=='moving':
            names+=['simulation_aggregates','aggregate_summary','paired_aggregate_summary','paired_aggregate_differences']
        for key in names:
            compressed=key.endswith('_differences') or key=='simulation_aggregates'
            data[key].to_csv(folder/(key+('.csv.gz' if compressed else '.csv')),index=False,
                compression={'method':'gzip','mtime':0} if compressed else None)
        artifacts={path.relative_to(root).as_posix():sha(path) for path in sorted(folder.iterdir())
                   if path.is_file() and path.name!='manifest.json'}
        manifest=dict(schema=1,status='completed',example='beamforming_comparison_audit',case=case,
            component='existing_data',config=data['config'],
            producer='scripts/summarize_beamforming_comparison_audit.py',
            sources={name:sha(root/name) for name in (
                'scripts/summarize_beamforming_comparison_audit.py',
                'lib/beamforming_comparison_summaries.py','lib/beamforming_moving_evidence.py')},
            original_sources=data['source_hashes'],inputs=data['input_hashes'],
            artifacts=artifacts,checks=data['checks'],
            scope='Read-only existing-data comparison. Every original observation, setting, inference result and previous export is preserved.')
        json_file('manifest.json',manifest)
        results[case]=dict(manifest=(folder/'manifest.json').relative_to(root).as_posix(),
            artifacts=len(artifacts),**data['checks'])
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    print(json.dumps(export(args.root),indent=2))


if __name__=='__main__':main()

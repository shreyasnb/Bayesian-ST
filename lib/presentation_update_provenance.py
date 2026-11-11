"""Verify explicit presentation updates without weakening data preservation.

The original preservation maps remain immutable. A later user-authorized plot
revision may change only the specific presentation files recorded in its proof;
both their prior and resulting hashes are checked. Scientific inputs, selected
settings, benchmark implementations and saved numerical tables are ineligible.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


PROOF = 'results/application_plotting_update/authorized_updates.json'
FOLLOWUP_PROOF = 'results/solid_trajectory_update/authorized_updates.json'
WIDTH_PROOF = 'results/moving_linewidth_update/authorized_updates.json'
NOTEBOOKS = {
    'notebooks/01_array.ipynb', 'notebooks/03_contact.ipynb',
    'notebooks/beamforming_moving_interference.ipynb',
    'notebooks/contact_prediction_diagnostics.ipynb',
    'notebooks/beamforming_moving_comparison_audit.ipynb',
    'notebooks/beamforming_stationary_comparison_audit.ipynb',
}
METADATA = {
    'results/notebook_editability.json', 'results/draft_figure_manifest.json',
    'results/main/README.md', 'results/supplementary/README.md',
    'results/beamforming_moving/figure_manifest.json',
    'results/beamforming_moving/preservation_baseline.json',
    'results/contact_prediction_diagnostics/figure_manifest.json',
    'results/contact_prediction_diagnostics/pdf_layout_checks.json',
    'results/contact_prediction_diagnostics/validation/independent_checks.json',
    'results/beamforming_moving/plotting/manifest.json',
    'results/beamforming_moving/plotting/run_guard.json',
    'results/beamforming_comparison_audit/moving/figure_manifest.json',
    'results/beamforming_comparison_audit/stationary/figure_manifest.json',
}
PURE_SPHERES = {
    'posterior_anisotropy_spheres.pdf', 'posterior_anisotropy_spheres.png',
    'posterior_anisotropy_sphere.pdf', 'posterior_anisotropy_sphere.png',
    'moving_sphere_posterior_spheres.pdf', 'moving_sphere_posterior_spheres.png',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def presentation_role(name):
    """Classify only notebook views, individual figure exports and metadata."""
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts:
        return None
    if name in NOTEBOOKS:
        return 'notebook'
    if name in METADATA:
        return 'presentation_metadata'
    if name == 'lib/presentation_update_provenance.py':
        return 'presentation_validation_source'
    if path.name in PURE_SPHERES:
        return None
    if (path.name in {'noise_adaptation_supp_page3.pdf', 'noise_adaptation_supp_page3.png'}
            and any(part in {'contact', '03_contact'} for part in path.parts)):
        return None
    if path.suffix not in ('.pdf', '.png'):
        return None
    prefix = '/'.join(path.parts[:-1])
    destinations = {
        'results/main/beamforming', 'results/main/contact',
        'results/supplementary/beamforming', 'results/supplementary/contact',
        'results/notebook_exports/01_array', 'results/notebook_exports/03_contact',
        'results/beamforming_moving/figures',
        'results/contact_prediction_diagnostics/figures',
        'results/beamforming_comparison_audit/moving/figures',
        'results/beamforming_comparison_audit/stationary/figures',
    }
    if prefix in destinations:
        # Complete historical catalogues are not rewritten by notebook saves.
        if path.name in {
            'setup.pdf', 'task_performance.pdf', 'posterior_anisotropy.pdf',
            'noise_adaptation.pdf', 'information.pdf', 'posterior_curvature.pdf',
            'tracking_bound.pdf', 'block_updates.pdf', 'additional_comparisons.pdf',
            'variance_estimation.pdf', 'predictive_calibration.pdf',
            'predictive_scores.pdf', 'model_mismatch.pdf', 'computation_time.pdf',
            'numerical_accuracy.pdf', 'paper_summary.pdf',
        } and prefix.startswith('results/supplementary/'):
            return None
        return path.suffix[1:]
    return None


def load_authorized_updates(root):
    """Resolve successive exact-hash revisions while retaining prior proofs.

    Every new revision must start at the previous revision's resulting hash.
    Only the final hash is required on disk. No intermediate proof is rewritten
    to pretend that an earlier audit observed the latest presentation.
    """
    root = Path(root).resolve()
    updates = {}
    for filename in (PROOF, FOLLOWUP_PROOF, WIDTH_PROOF):
        path = root/filename
        if not path.is_file():
            continue
        proof = json.loads(path.read_text())
        if (proof.get('schema') != 1 or proof.get('status') != 'verified'
                or proof.get('scope') != 'User-requested application plotting updates'):
            raise RuntimeError('Unknown application presentation update proof')
        for name, record in proof['updates'].items():
            role = presentation_role(name)
            if role is None or role != record.get('role'):
                raise RuntimeError('Scientific or unapproved file in plotting update proof: '+name)
            if not (isinstance(record.get('before_sha256'), str)
                    and len(record['before_sha256']) == 64
                    and isinstance(record.get('after_sha256'), str)
                    and len(record['after_sha256']) == 64):
                raise RuntimeError('Presentation update needs both actual file hashes: '+name)
            previous = updates.get(name)
            if previous is not None and previous['after_sha256'] != record['before_sha256']:
                raise RuntimeError('Disconnected presentation revision: '+name)
            prior_hashes = previous['permitted_before_sha256'][:] if previous else []
            prior_hashes.append(record['before_sha256'])
            updates[name] = dict(record, permitted_before_sha256=prior_hashes)
    for name, record in updates.items():
        if not (root/name).is_file() or sha(root/name) != record['after_sha256']:
            raise RuntimeError('Changed presentation update output: '+name)
    return updates


def verify_presentation_preservation(root, mapping):
    """Check every prior file, allowing only exact authorized plot revisions."""
    root = Path(root).resolve()
    updates = None
    authorized = []
    unchanged = 0
    for name, record in mapping.items():
        expected = record['sha256'] if isinstance(record, dict) else record
        path = root/name
        if path.is_file() and sha(path) == expected:
            unchanged += 1
            continue
        if updates is None:
            updates = load_authorized_updates(root)
        approved = updates.get(name)
        if (approved is None or expected not in approved['permitted_before_sha256']
                or not path.is_file() or sha(path) != approved['after_sha256']):
            raise RuntimeError('Unapproved change to preserved evidence: '+name)
        authorized.append(name)
    return dict(status='passed', files_checked=len(mapping), unchanged_files=unchanged,
                authorized_presentation_updates=sorted(authorized),
                proof=PROOF if authorized else None,
                followup_proof=FOLLOWUP_PROOF if authorized and (root/FOLLOWUP_PROOF).is_file() else None)

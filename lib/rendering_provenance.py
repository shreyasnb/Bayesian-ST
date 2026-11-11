"""Resolve recorded rendering sources without rebinding historical outputs.

A retained figure continues to name its actual producer bytes. If a source has
changed, its recorded digest is resolved to a verified immutable copy, rather
than being replaced by the digest of the current implementation. No numerical
artifact, validation certificate, PDF or preview is modified by these helpers.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess


SOURCE_REVISION = 'dd399bacae3d7b37e31abb949b24740c1f07a4f3'
ARCHIVE_ROOT = Path('archive/dd399ba/rendering_sources')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(record):
    return record['sha256'] if isinstance(record, dict) else record


def _relative(name):
    path = PurePosixPath(str(name))
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Rendering source paths must be repository-relative: '+str(name))
    return Path(*path.parts)


def _merge(target, mapping):
    for name, expected in mapping.items():
        if name in target and _digest(target[name]) != _digest(expected):
            raise RuntimeError('Conflicting rendering-source claims for '+name)
        target[name] = expected
    return target


def resolve_source_mapping(root, mapping, revision=SOURCE_REVISION):
    """Return verifiable paths for the exact digests in ``mapping``.

    The current file is used only if it already has the recorded bytes. The
    scientific source archive is reused when available. Other changed sources
    are copied from the specified immutable Git revision only after their full
    SHA256 equals the recorded digest. Existing archive files are never replaced.
    """
    root = Path(root).resolve()
    proof_path = root/'archive/dd399ba/stationary_source_proof.json'
    proof = json.loads(proof_path.read_text()) if proof_path.is_file() else {}
    scientific = proof.get('source_files', {})
    resolved = {}
    for original_name, record in mapping.items():
        original = _relative(original_name)
        expected = _digest(record)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError('Rendering source needs a SHA256 digest: '+str(original_name))
        current = root/original
        if current.is_file() and _sha(current) == expected:
            _merge(resolved, {original.as_posix(): deepcopy(record)})
            continue
        archived_record = scientific.get(original.as_posix(), {})
        if archived_record.get('sha256') == expected:
            archive = _relative(archived_record['archive_path'])
            if not (root/archive).is_file() or _sha(root/archive) != expected:
                raise RuntimeError('Historical scientific source is missing or changed: '+str(archive))
            _merge(resolved, {archive.as_posix(): deepcopy(record)})
            continue
        archive = ARCHIVE_ROOT/original
        if (root/archive).exists() and _sha(root/archive) != expected:
            archive = ARCHIVE_ROOT/expected/original
        if (root/archive).exists():
            if not (root/archive).is_file() or _sha(root/archive) != expected:
                raise RuntimeError('Historical rendering source changed: '+str(archive))
        else:
            retrieved = subprocess.run(
                ['git', 'show', revision+':'+original.as_posix()], cwd=root,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if retrieved.returncode:
                raise RuntimeError('Cannot retrieve recorded rendering source '+str(original)
                                   +' from '+revision+': '+retrieved.stderr.decode(errors='replace').strip())
            actual = hashlib.sha256(retrieved.stdout).hexdigest()
            if actual != expected:
                raise RuntimeError('Recorded rendering source digest does not match '+revision+': '+str(original))
            destination = root/archive
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation preserves a pre-existing archive during races.
            try:
                with destination.open('xb') as output:
                    output.write(retrieved.stdout)
            except FileExistsError:
                pass
            if not destination.is_file() or _sha(destination) != expected:
                raise RuntimeError('Archived rendering source digest disagrees: '+str(archive))
        _merge(resolved, {archive.as_posix(): deepcopy(record)})
    return resolved


def archive_manifest_provenance(root, meta, revision=SOURCE_REVISION):
    """Copy a figure manifest with verifiable historical producer source maps.

    Only ``sources`` and each figure's ``source_hashes`` change. Their digest
    values remain the recorded values. All data, gate, file, page and output
    records are preserved, and no manifest is written to disk.
    """
    result = deepcopy(meta)
    sources = resolve_source_mapping(root, result.get('sources', {}), revision)
    for figure in result.get('figures', {}).values():
        if 'source_hashes' in figure:
            figure['source_hashes'] = resolve_source_mapping(root, figure['source_hashes'], revision)
            _merge(sources, figure['source_hashes'])
    result['sources'] = sources
    return result

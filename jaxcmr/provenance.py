"""Capture notebook parameters, input content and the actual local source."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil

from jaxcmr.helpers import resolve_project_path


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def capture_notebook(parameters, output_dir, source_paths=(), notebook_source=None, *, base_dir=None):
    """Snapshot source and existing parameter-referenced input files.

    Runs remain local. Repeating an identical stage reuses its content-addressed
    capture; differing parameters or source produce a separate immutable capture.
    """
    base = resolve_project_path(base_dir, Path.cwd()) if base_dir is not None else Path.cwd()
    parameters = _json_value(parameters)
    inputs = {}

    def visit(value):
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str) and value:
            try:
                path = resolve_project_path(value, base)
                if path.is_file():
                    inputs[str(path.resolve())] = file_sha256(path)
            except (OSError, ValueError):
                pass

    visit(parameters)
    sources, source_files = {}, {}
    for index, source_path in enumerate(source_paths):
        root = resolve_project_path(source_path, base)
        candidates = sorted(root.rglob('*.py')) if root.is_dir() else [root]
        for path in candidates:
            if not path.is_file():
                continue
            relative = path.relative_to(root) if root.is_dir() else Path(path.name)
            name = f'{index}_{root.name}/{relative.as_posix()}'
            sources[name] = file_sha256(path)
            source_files[name] = path
    notebook_path = resolve_project_path(notebook_source, base) if notebook_source else None
    if notebook_path is not None and notebook_path.is_file():
        path = notebook_path
        sources['notebook/' + path.name] = file_sha256(path)
        source_files['notebook/' + path.name] = path
    source_hash = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    record = dict(parameters=parameters, input_sha256=inputs, source_sha256=sources,
                  source_fingerprint=source_hash)
    identity = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    destination = resolve_project_path(output_dir, base) / identity[:16]
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / 'manifest.json'
    if not manifest.exists():
        for relative, source in source_files.items():
            target = destination / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        versions = {'python': platform.python_version()}
        for name in ['jaxcmr', 'jax', 'jaxlib', 'numpy', 'scipy', 'h5py', 'matplotlib']:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = 'local source'
        record.update(identity=identity, versions=versions,
                      created_utc=datetime.now(timezone.utc).isoformat())
        manifest.write_text(json.dumps(record, indent=2) + '\n')
    return dict(path=str(manifest.resolve()), identity=identity, source_fingerprint=source_hash)

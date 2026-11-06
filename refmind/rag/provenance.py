"""Content/configuration identities carried from parsing to indexed evidence."""
from __future__ import annotations
import hashlib
import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from ..config import settings


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _configuration_hash(configuration: dict) -> str:
    canonical = json.dumps(configuration, ensure_ascii=True, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def parsing_provenance(parsed: dict) -> dict:
    """Retain the parser that actually produced cached text; unknown stays unknown."""
    explicit = parsed.get('parsing_provenance')
    legacy = parsed.get('processing_versions')
    recorded = isinstance(explicit, dict) or isinstance(legacy, dict)
    snapshot = explicit if isinstance(explicit, dict) else legacy if isinstance(legacy, dict) else {}
    # A legacy processing snapshot also contains chunker/embedding settings;
    # those describe its previous operation, not the parsing event itself.
    result = dict(explicit) if isinstance(explicit, dict) else {
        key: snapshot[key] for key in ('parser_version', 'parser_options', 'parser_configuration') if key in snapshot}
    result.pop('provenance_sha256', None)
    packages = snapshot.get('packages') if isinstance(snapshot.get('packages'), dict) else {}
    result.update({'schema': 1, 'parser': str(snapshot.get('parser') or parsed.get('parser') or ''),
                   'packages': {str(key): value for key, value in packages.items()
                                if key != 'langchain-text-splitters'},
                   'recorded': bool(snapshot.get('recorded', recorded))})
    if parsed.get('schema_version') is not None:
        result.setdefault('source_schema_version', parsed['schema_version'])
    return {**result, 'provenance_sha256': _configuration_hash(result)}


def processing_versions(parser: str, *, parsing: dict | None = None) -> dict:
    packages = {}
    for package in ('mineru', 'PyMuPDF', 'langchain-text-splitters'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = 'not_installed'
    if parsing is not None:
        # Rechunking uses the current splitter and embedding configuration while
        # its source text retains the original parser/package identity.
        packages = {**dict(parsing.get('packages') or {}),
                    'langchain-text-splitters': packages['langchain-text-splitters']}
        parser = str(parsing.get('parser') or parser)
    provider = os.getenv('EMBEDDING_PROVIDER', 'dashscope')
    configuration = {'schema': 1, 'parser': parser, 'packages': packages,
                     'chunker': 'layout-v3', 'chunk_size': settings.chunk_size,
                     'chunk_overlap': settings.chunk_overlap,
                     'layout_chunk_max_chars': settings.layout_chunk_max_chars,
                     'embedding_provider': provider,
                     'embedding_model': 'text-embedding-3-small' if provider == 'openai' else settings.embedding_model}
    if parsing is not None:
        configuration['parsing_provenance'] = parsing
    return {**configuration, 'configuration_sha256': _configuration_hash(configuration)}

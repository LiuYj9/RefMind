"""Offline, deterministic before/after evaluation of chunk evidence integrity."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from refmind.rag import layout_chunker as current

OUT = ROOT / 'evaluation/reports/chunking_20261103'
spec = importlib.util.spec_from_file_location('chunk_baseline', OUT / 'layout_chunker_before.py')
baseline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = baseline
spec.loader.exec_module(baseline)


def b(identity, text, kind='paragraph', page=1, end=None):
    return dict(block_id=identity, text=text, type=kind, page=page, page_end=end or page)


atomic_cases = [
    [b('t', 'Table: short\n|A|1|', 'table')],
    [b('t', '<table>' + '<tr><td>10</td><td></td></tr>' * 100 + '</table>', 'table', 2, 3)],
    [b('t', 'Table: values\n' + '|A|1|\n' * 1000, 'table')],
    [b('e', r'E=mc^2', 'equation')],
    [b('e', r'\begin{aligned}' + r'x_i &= y_i + z_i \\' * 200 + r'\end{aligned}', 'equation')],
    [b('e', 'x' * 2000, 'equation')],
]
association_cases = [
    ([b('e', 'E=mc^2', 'equation'), b('d', 'where m is mass.')], True),
    ([b('e', 'E=mc^2', 'equation'), b('d', '其中 m 为质量。')], True),
    ([b('e', 'E=mc^2', 'equation', 2), b('d', '式中 m 为质量。', page=3)], True),
    ([b('e', 'E=mc^2', 'equation'), b('d', 'The experiment improved.')], False),
    ([b('e', 'E=mc^2', 'equation'), b('h', 'Methods', 'heading'), b('d', 'where m is mass.')], False),
    ([b('e', 'E=mc^2', 'equation'), b('d', 'where m is mass.', page=3)], False),
    ([b('e', 'E=mc^2', 'equation'), b('d', 'where ' + 'm' * 700)], False),
]
page_cases = [
    [b('a', 'A' * 400, page=2), b('b', 'B' * 400, page=3)],
    [b('a', 'A' * 200, page=2), b('b', 'B' * 200, page=3), b('c', 'C' * 200, page=4)],
    [b('a', 'Repeated paragraph.', page=2), b('b', 'Repeated paragraph.', page=3)],
    [b('a', 'A' * 2500, page=2), b('b', 'B' * 2500, page=3)],
]


def evaluate(module):
    split = lambda blocks: module.layout_chunks(blocks, 1000, 1800, 200)
    # Existing parser/chunker contract strips outer whitespace before indexing.
    atomic = [len(out := split(case)) == 1 and out[0].text == case[0]['text'].strip() for case in atomic_cases]
    associations = []
    for case, expected in association_cases:
        out = split(case)
        linked = any('e' in c.block_ids and 'd' in c.block_ids for c in out)
        associations.append(dict(expected=expected, linked=linked, passed=linked == expected))
    pages = [all(c.page_start == c.page_end for c in split(case)) for case in page_cases]
    return dict(atomic=dict(passed=sum(atomic), total=len(atomic), cases=atomic),
                association=dict(passed=sum(c['passed'] for c in associations), total=len(associations), cases=associations),
                page_localization=dict(passed=sum(pages), total=len(pages), cases=pages))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


dev_dir = ROOT / 'evaluation/workspaces/phase1-openai-v1/parsed'
files = sorted(dev_dir.glob('*.json'))
if len(files) != 7:
    raise ValueError('Expected exactly the seven previously approved dev snapshots')
dev = []
for path in files:
    parsed = json.loads(path.read_text(encoding='utf-8'))
    blocks = parsed.get('blocks') or []
    row = dict(file=path.name, sha256=digest(path), parser=parsed.get('parser'),
               input_blocks=len(blocks), table_blocks=sum(b.get('type') == 'table' for b in blocks),
               equation_blocks=sum(b.get('type') == 'equation' for b in blocks))
    for name, module in (('before', baseline), ('after', current)):
        chunks = module.layout_chunks(blocks, 1000, 1800, 200)
        row[name] = dict(chunks=len(chunks), chars=sum(len(c.text) for c in chunks),
                         cross_page_chunks=sum(c.page_start != c.page_end for c in chunks),
                         max_chars=max((len(c.text) for c in chunks), default=0))
    dev.append(row)

result = dict(parameters=dict(target_chars=1000, max_chars=1800, overlap=200),
              synthetic=dict(before=evaluate(baseline), after=evaluate(current)), dev=dev,
              source_hashes={p.name: digest(p) for p in (OUT / 'layout_chunker_before.py',
                            ROOT / 'refmind/rag/layout_chunker.py', ROOT / 'refmind/rag/document_processor.py')},
              limitations=['Synthetic labels are authored engineering expectations, not human quality certification.',
                           'Seven saved dev parses are PyMuPDF, with no identified table/equation blocks.',
                           'No paid calls, holdout evaluation, vector rebuild, or end-to-end QA measurement.'])
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(dict(synthetic=result['synthetic'],
                     dev_chunks={k: sum(r[k]['chunks'] for r in dev) for k in ('before', 'after')}), ensure_ascii=False))
assert all(metric['passed'] == metric['total'] for metric in result['synthetic']['after'].values())

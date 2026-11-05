"""Plumbing test for bench/bench_pattern_dino.py with the fake dense model on a 8-image synthetic corpus
(NOT a benchmark: it only proves the script wires cases -> cells -> projection -> variants -> fusion ->
report, and that --fit --write-proj writes a loadable projection). Real numbers need the GPU machine."""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'bench'))

import bench_pattern_dino as bp  # noqa: E402
from media_manager import pattern_dino as pd  # noqa: E402
from test_pattern_dino import checker, make_embedder, stripes  # noqa: E402


def _corpus(tmp_path):
    rng = np.random.default_rng(2)
    d = tmp_path / 'data'
    d.mkdir()
    imgs = {'q.png': stripes(rng), 's1.png': stripes(rng), 's2.png': stripes(rng, period=10),
            'c1.png': checker(rng), 'c2.png': checker(rng, cell=6),
            'n.png': rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
            'sm.png': np.full((256, 256, 3), (40, 120, 200), np.uint8),
            'q2.png': checker(rng)}
    for n, a in imgs.items():
        Image.fromarray(a).save(d / n)
    cases = [
        {'id': 'p1', 'task': 'pattern', 'synthetic': False, 'query_image': 'q.png', 'bbox': [10, 10, 130, 130],
         'positives': ['s1.png', 's2.png'], 'hard_negatives': ['c1.png']},
        {'id': 'p2', 'task': 'pattern', 'synthetic': True, 'query_image': 'q2.png', 'bbox': [20, 20, 140, 140],
         'positives': ['c1.png', 'c2.png'], 'hard_negatives': ['s1.png']},
        {'id': 'r1', 'task': 'region', 'synthetic': False, 'query_image': 'q.png', 'bbox': [0, 0, 50, 50],
         'positives': ['s1.png'], 'hard_negatives': []},       # other tasks are ignored
    ]
    (d / 'manifest.json').write_text(json.dumps({'cases': cases}))
    return d


def test_bench_runs_end_to_end_with_fake_model(tmp_path, monkeypatch):
    monkeypatch.setattr(bp, 'make_embedder', make_embedder)
    pd.clear_projection()
    d = _corpus(tmp_path)
    out = tmp_path / 'report.json'
    rep = bp.main(['--data-dir', str(d), '--dims', '8,16', '--skip-resnet',
                   '--cache-dir', str(tmp_path / 'cache'), '--json', str(out)])
    assert rep['n_cases'] == 2 and rep['corpus_images'] == 8
    assert any(k.startswith('dino pca16 both') for k in rep['dino'])
    assert 'dino noproj(1024) both' in rep['dino']
    best = max(v['metrics']['all']['ap'] for v in rep['dino'].values())
    assert best > 0.6, rep['dino']                     # the fake model separates stripes from checker
    assert json.loads(out.read_text())['cost']['img_per_s'] > 0
    # second run reuses the cached raw cells (no new forwards)
    emb = make_embedder()
    monkeypatch.setattr(bp, 'make_embedder', lambda: emb)
    bp.main(['--data-dir', str(d), '--dims', '8', '--skip-resnet', '--no-oracle',
             '--cache-dir', str(tmp_path / 'cache')])
    assert emb.model.shapes and all(s[0] == 1 for s in emb.model.shapes)   # only batch-1 query forwards


def test_fit_and_write_projection(tmp_path, monkeypatch):
    monkeypatch.setattr(bp, 'make_embedder', make_embedder)
    monkeypatch.setattr(pd, '_PROJ_PATH', str(tmp_path / 'proj.npz'))
    pd.clear_projection()
    d = _corpus(tmp_path)
    bp.main(['--data-dir', str(d), '--fit', '--dim', '12', '--write-proj', '--cache-dir', str(tmp_path / 'cache')])
    assert os.path.isfile(tmp_path / 'proj.npz')
    pd.clear_projection()                                # reload from the written file
    assert pd.projection_ready() and pd.dim() == 12
    pd.clear_projection()


def test_fusion_branch_with_stubbed_resnet(tmp_path, monkeypatch):
    """The fusion/ResNet half of the report (normally needs timm + the GPU) with stand-in windows."""
    from media_manager import pattern_deep as pdp
    monkeypatch.setattr(bp, 'make_embedder', make_embedder)
    pd.clear_projection()

    def fake_windows(rgb):
        v = np.zeros(pdp.DIM, np.float32)
        v[int(rgb.reshape(-1, 3).std() // 12) % pdp.DIM] = 1.0
        return pdp.quantize(np.stack([v] * 3))
    monkeypatch.setattr(pdp, 'describe_image', fake_windows)
    monkeypatch.setattr(pdp, 'describe_query', lambda rgb, box: fake_windows(
        rgb[int(box[1]):int(box[3]), int(box[0]):int(box[2])])[:1].astype(np.float32) / 127.0)
    d = _corpus(tmp_path)
    rep = bp.main(['--data-dir', str(d), '--dims', '8', '--no-oracle', '--cache-dir', str(tmp_path / 'cache')])
    assert 'resnet' in rep and rep['resnet']['metrics']['all']['n'] == 2
    assert len(rep['fusion']) == 9                       # 3 best DINO variants x 3 weights
    assert all(0 <= v['metrics']['all']['ap'] <= 1 for v in rep['fusion'].values())

"""scripts/eval_location_place.py core on a tiny fake library (no model, no GPU)."""
import importlib.util
import os

import numpy as np
import pytest

from media_manager import location_place as lp
from media_manager.database import Database
from media_manager.manual_db import ManualDB

_spec = importlib.util.spec_from_file_location(
    'eval_location_place', os.path.join(os.path.dirname(__file__), '..', 'scripts', 'eval_location_place.py'))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def _vec(dim, **c):
    v = np.zeros(dim, np.float32)
    for k, x in c.items():
        v[int(k[1:])] = x
    return (v / np.linalg.norm(v)).astype(np.float32)


@pytest.fixture
def lib(tmp_path):
    os.makedirs(tmp_path / '.media')
    db = Database(str(tmp_path / '.media' / 'media.db'))
    manual = ManualDB(str(tmp_path / '.media' / 'manual.db'))
    ids = {}

    def add(name, **e):
        cs = (name * 12)[:40].ljust(40, 'x')
        fid = db.upsert_file_path(f'{name}.jpg', cs, size=1)
        db.insert_siglip_embedding(fid, _vec(10, **e).tobytes(), 's')
        db.insert_dino_embedding(fid, _vec(8, **e).tobytes(), 'd')
        ids[name] = (fid, cs)
    for i in range(10):
        add(f'a{i}', e0=1, **{f'e{1 + i % 2}': 0.03})
        add(f'b{i}', e4=1, **{f'e{5 + i % 2}': 0.03})
    for i in range(6):
        add(f'bait{i}', e0=1, e3=1)          # looks only vaguely like A
    for i in range(6):
        add(f'fill{i}', e7=1)
    db.conn.commit()
    la, lb, lbait = manual.create_location('A'), manual.create_location('B'), manual.create_location('Elsewhere')
    for i in range(10):
        manual.add_file_location(ids[f'a{i}'][1], la)
        manual.add_file_location(ids[f'b{i}'][1], lb)
        manual.assign_identity_to_photo(ids[f'a{i}'][1], 'Ann')
    for i in range(6):
        manual.add_file_location(ids[f'bait{i}'][1], lbait)
        manual.assign_identity_to_photo(ids[f'bait{i}'][1], 'Ann')
    db.conn.commit()
    return db, manual, ids


def test_evaluate_core_and_bait_probe(lib, monkeypatch):
    db, manual, _ids = lib
    monkeypatch.setattr(lp, 'PEOPLE_LAMBDA', 50.0)       # make the damping decisive on a 38-photo toy library
    res = ev.evaluate(db, manual, min_members=8, holdout=0.2, seed=1, k_prec=5, k_rec=8)
    assert [e['name'] for e in res['per_location']] == ['A', 'B']        # 'Elsewhere' has 6 < 8 members
    sm = res['summary']
    assert set(sm) == {'siglip', 'dino', 'combined', 'place'}
    for v in sm.values():
        assert v['locations'] == 2 and 0 <= v['p_at_20'] <= 1 and v['r_at_100'] == pytest.approx(1.0)
        assert v['n_probe'] == 1                                         # only A has a recurring identity
    assert sm['combined']['p_at_20'] == pytest.approx(0.4) and sm['place']['p_at_20'] == pytest.approx(0.4)
    a = res['per_location'][0]['variants']
    assert a['combined']['bait_at_20'] == 3 and a['place']['bait_at_20'] == 0     # bait displaced by damping
    assert res['per_location'][1]['variants']['place']['bait_at_20'] is None
    again = ev.evaluate(db, manual, min_members=8, holdout=0.2, seed=1, k_prec=5, k_rec=8)
    assert again['summary'] == res['summary']                                # deterministic per seed
    table = ev.format_table(res, ['siglip', 'dino', 'combined', 'place'])
    assert 'P@20' in table and 'bait@20' in table and 'place' in table


def test_cli_main(lib, tmp_path, capsys):
    db, _m, _i = lib
    db.conn.commit()
    out = tmp_path / 'r.json'
    assert ev.main(['--data-root', str(tmp_path), '-q', '--variants', 'combined', 'place', '--json', str(out)]) == 0
    assert out.exists() and 'combined' in capsys.readouterr().out

"""Unit tests for media_manager.face_clusters (pure numpy)."""
import numpy as np
import pytest

from media_manager.face_clusters import GroupSearchCancelled, find_face_groups

D = 32


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _cluster(rng, center, n, spread):
    return _unit(center + spread * rng.standard_normal((n, center.shape[0])).astype(np.float32))


def _planted(seed=0, sizes=(40, 25, 10), noise=60, spread=0.08):
    rng = np.random.default_rng(seed)
    centers = _unit(rng.standard_normal((len(sizes), D)))
    parts, labels = [], []
    for k, (c, sz) in enumerate(zip(centers, sizes)):
        parts.append(_cluster(rng, c, sz, spread))
        labels += [k] * sz
    parts.append(_unit(rng.standard_normal((noise, D))))
    labels += [-1] * noise
    x = np.concatenate(parts).astype(np.float32)
    return np.arange(1000, 1000 + len(x)), x, np.array(labels)


def test_planted_clusters_recovered_biggest_first():
    ids, x, labels = _planted()
    groups = find_face_groups(ids, x, threshold=0.6, min_size=3, chunk=37)
    sizes = [len(g['face_ids']) for g in groups]
    assert sizes[:3] == [40, 25, 10]
    assert sizes == sorted(sizes, reverse=True)
    for g, k in zip(groups[:3], (0, 1, 2)):
        lab = labels[np.array(g['face_ids']) - 1000]
        assert set(lab) == {k}


def test_members_sorted_and_all_above_threshold_to_centroid():
    ids, x, _ = _planted(seed=3)
    thr = 0.6
    for g in find_face_groups(ids, x, threshold=thr, min_size=3):
        assert g['scores'] == sorted(g['scores'], reverse=True)
        pos = np.array(g['face_ids']) - 1000
        sims = x[pos] @ g['centroid']
        assert np.allclose(sims, g['scores'], atol=1e-5)
        assert (sims >= thr).all()
        assert abs(np.linalg.norm(g['centroid']) - 1) < 1e-4


def test_groups_are_disjoint():
    ids, x, _ = _planted(seed=5)
    seen = set()
    for g in find_face_groups(ids, x, threshold=0.5, min_size=2):
        assert not seen & set(g['face_ids'])
        seen |= set(g['face_ids'])


def test_chain_is_not_single_linkage():
    # points walking along an arc, adjacent ones 0.9 similar: single linkage at 0.9
    # would merge all 12 into one group; the centroid check must cut the chain.
    step = np.arccos(0.93)
    pts = np.array([[np.cos(i * step), np.sin(i * step)] + [0] * (D - 2) for i in range(12)],
                   dtype=np.float32)
    thr = 0.9
    groups = find_face_groups(np.arange(12), pts, threshold=thr, min_size=2)
    assert groups
    assert max(len(g['face_ids']) for g in groups) < 12
    for g in groups:
        assert (pts[g['face_ids']] @ g['centroid'] >= thr - 1e-6).all()
        # the two ends of the chain never share a group
        assert not ({0, 11} <= set(g['face_ids']))


def test_threshold_edge():
    a = np.zeros(D, np.float32); a[0] = 1
    def at(sim):
        v = np.zeros(D, np.float32); v[0] = sim; v[1] = np.sqrt(1 - sim ** 2); return v
    x = np.stack([a] * 8 + [at(0.80), at(0.60)])
    g = find_face_groups(np.arange(10), x, threshold=0.75, min_size=3)
    assert len(g) == 1
    assert 9 not in g[0]['face_ids']          # 0.60 < 0.75 stays out
    assert 8 in g[0]['face_ids']              # 0.80 >= 0.75 comes in
    assert set(range(8)).issubset(g[0]['face_ids'])


def test_min_size_respected():
    ids, x, _ = _planted(sizes=(5, 2), noise=20)
    for ms in (3, 6):
        for g in find_face_groups(ids, x, threshold=0.6, min_size=ms):
            assert len(g['face_ids']) >= ms
    assert find_face_groups(ids, x, threshold=0.6, min_size=60) == []


def test_max_groups_and_empty():
    ids, x, _ = _planted()
    assert len(find_face_groups(ids, x, threshold=0.6, max_groups=1)) == 1
    assert find_face_groups([], np.zeros((0, D), np.float32)) == []


def test_deterministic_and_chunk_independent():
    ids, x, _ = _planted(seed=7)
    a = find_face_groups(ids, x, threshold=0.55, chunk=17)
    b = find_face_groups(ids, x, threshold=0.55, chunk=4096)
    assert [g['face_ids'] for g in a] == [g['face_ids'] for g in b]
    c = find_face_groups(ids, x, threshold=0.55, chunk=17)
    assert [g['face_ids'] for g in a] == [g['face_ids'] for g in c]


def test_gemm_hook_used_and_cpu_none_fallback():
    ids, x, _ = _planted(seed=2)
    calls = []
    def gemm(a, b):
        calls.append(a.shape)
        return a @ b
    g1 = find_face_groups(ids, x, threshold=0.6, gemm=gemm, chunk=50)
    g2 = find_face_groups(ids, x, threshold=0.6, gemm=lambda a, b: None, chunk=50)
    assert calls
    assert [g['face_ids'] for g in g1] == [g['face_ids'] for g in g2]


def test_progress_and_cancel():
    ids, x, _ = _planted()
    seen = []
    find_face_groups(ids, x, threshold=0.6, chunk=40, progress=lambda d, t: seen.append((d, t)))
    assert seen[-1][0] == seen[-1][1]
    assert all(b >= a for (a, _), (b, _) in zip(seen, seen[1:]))
    n = {'k': 0}
    def stop():
        n['k'] += 1
        return n['k'] > 2
    with pytest.raises(GroupSearchCancelled):
        find_face_groups(ids, x, threshold=0.6, chunk=20, should_stop=stop)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        find_face_groups([1, 2], np.zeros((3, D), np.float32))

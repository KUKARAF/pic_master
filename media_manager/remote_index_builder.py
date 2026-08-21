"""Host-side driver that (re)builds the worker's in-memory search index ("imdb").

Streams the CLIP + face embedding matrices from media.db to the worker in bounded
chunks, so this low-RAM host NEVER materializes the full ~1 GB matrix (that OOM is
the whole reason the offload exists) — it reads the DB with a fetchmany cursor and
ships each chunk as it goes (see Database.iter_clip_embeddings / iter_face_embeddings).

It also TIMES each kind, so we can see how long the RNS transfer of ~1 GB actually
takes — the number this milestone exists to measure. Rebuild is manual only (per
design): triggered from web.py's /api/imdb/rebuild in a background thread.

This process stays LIGHT — it only imports Database + worker_client, never
torch/ultralytics/insightface.
"""
import os
import time

# ~10000 rows × 512 float32 ≈ 20 MB per chunk — modest enough for one RNS request,
# big enough that the ~470 rows-per-MB overhead of many tiny requests doesn't dominate.
_CHUNK_ROWS = 10000


def _client(data_root):
    from media_manager import worker_client
    return worker_client.get_client(data_root)


def build_index(data_root, kinds=('clip', 'face'), log=print, progress=None):
    """Rebuild the named matrices on the worker by streaming them over RNS.

    Returns {kind: {'rows', 'seconds', 'mb', 'mbps'}}. `progress(kind, rows_sent,
    seconds)` is an optional callback for a web job's live status. Raises if the
    worker is unavailable (no local fallback — that would reload the matrix here and
    OOM, defeating the purpose)."""
    from media_manager.database import Database

    client = _client(data_root)
    if not client.is_available():
        raise RuntimeError('Worker unavailable — start/configure the media worker '
                           'before rebuilding the imdb index.')

    db = Database(os.path.join(os.path.abspath(data_root), '.media', 'media.db'))
    stats = db.embedding_stats()
    iters = {'clip': db.iter_clip_embeddings, 'face': db.iter_face_embeddings}

    results = {}
    for kind in kinds:
        if kind not in iters:
            raise ValueError(f'unknown imdb kind {kind!r}')
        dim = stats.get(kind, {}).get('dim', 0)
        rows_total = stats.get(kind, {}).get('rows', 0)
        if not dim or not rows_total:
            log(f'[imdb] {kind}: no embeddings to ship, skipping')
            results[kind] = {'rows': 0, 'seconds': 0.0, 'mb': 0.0, 'mbps': 0.0}
            continue

        log(f'[imdb] building {kind} matrix ({rows_total:,} rows × {dim}) ...')
        t0 = time.time()
        client.imdb_build_begin(kind, dim, total_rows=rows_total)
        sent = 0
        for ids, vecs in iters[kind](chunk=_CHUNK_ROWS):
            sent = client.imdb_build_chunk(kind, ids, vecs)
            if progress:
                progress(kind, sent, time.time() - t0)
        end = client.imdb_build_end(kind)
        secs = time.time() - t0
        mb = end.get('bytes', 0) / 1048576
        results[kind] = {'rows': end.get('rows', 0), 'seconds': round(secs, 1),
                         'mb': round(mb, 1), 'mbps': round(mb / secs, 2) if secs else 0.0}
        log(f'[imdb] {kind}: {end.get("rows", 0):,} rows / {mb:.1f} MB in {secs:.1f}s '
            f'({results[kind]["mbps"]} MB/s)')

    return results

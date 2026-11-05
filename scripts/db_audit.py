#!/usr/bin/env python3
"""Read-only audit of a media_manager data root, to decide which DB optimisations matter.

    python scripts/db_audit.py [--data-root PATH] [--quick] [--json OUT.json]
                               [--timeout SECONDS] [--runs N]

PATH is the directory that contains `.media/media.db` (and manual.db / error.db); it
defaults to the current directory.

STRICTLY READ-ONLY: every connection is opened with the `file:...?mode=ro` URI and
`PRAGMA query_only=ON`; nothing is created, altered, vacuumed or checkpointed, and nothing
is written into the data root. (Caveat inherent to SQLite: reading a WAL database needs
the `-shm` file; if the app is stopped and no -wal/-shm exist, SQLite may have nothing to
open in WAL mode read-only and the audit reports the DB as unreadable rather than
creating files. Run it while the app is up, or after a normal shutdown if that happens.)

Safe to run against production while the server is up, but it competes for I/O: --quick
skips the whole-table loads (the expensive part) and keeps sizes, plans and cheap timings.
"""
import argparse
import json
import os
import re
import sqlite3
import statistics
import sys
import time
import urllib.parse

MB = 1024 * 1024
GB = 1024 * MB

# Tables that hold per-file vectors / blobs worth singling out.
EMBEDDING_TABLES = ('embeddings', 'embeddings_siglip', 'embeddings_dino', 'tile_embeddings',
                    'body_embeddings', 'object_region_embeddings', 'place_embeddings', 'faces')
BLOB_FLAG_TABLES = ('embeddings', 'embeddings_siglip', 'embeddings_dino', 'tile_embeddings',
                    'body_embeddings', 'region_windows', 'pattern_windows')
# 'clip' = the legacy `embeddings` table: unused by the app (CLIP was removed) but audited as long as
# a user's DB still carries it.
SPACES = {'clip': 'embeddings', 'siglip': 'embeddings_siglip', 'dino': 'embeddings_dino'}


# --------------------------------------------------------------------------- plumbing
class Audit:
    def __init__(self, data_root, quick=False, timeout=60.0, runs=3, apply_app_pragmas=True):
        self.root = os.path.abspath(data_root)
        self.media_dir = os.path.join(self.root, '.media')
        self.quick = quick
        self.timeout = float(timeout)
        self.runs = max(1, int(runs))
        self.apply_app_pragmas = apply_app_pragmas
        self.deadline = None
        self.conns = {}
        self.R = {'data_root': self.root, 'quick': quick, 'timeout_s': self.timeout,
                  'runs': self.runs, 'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                  'sections': {}}
        self.findings = []

    # -- connections -------------------------------------------------------------
    def open_ro(self, path, tune=False):
        uri = 'file:' + urllib.parse.quote(os.path.abspath(path)) + '?mode=' + self._open_mode(path)
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        conn.execute('PRAGMA query_only=ON')
        if tune and self.apply_app_pragmas:
            # Same per-connection tuning the app uses (read-only pragmas, no file writes),
            # so timings reflect how the server actually reads.
            conn.execute('PRAGMA cache_size=-%d' % (128 * 1024))
            conn.execute('PRAGMA mmap_size=%d' % (4096 * MB))
            conn.execute('PRAGMA temp_store=MEMORY')
        conn.set_progress_handler(self._progress, 5000)
        return conn

    def _open_mode(self, path):
        """'ro' normally. A plain read-only open of a WAL database CREATES the -wal/-shm
        side files when they are absent (e.g. the app is stopped) - a write into the data
        root. In that case there is no live WAL state to honour, so open with immutable=1
        instead (reads the main file only, touches nothing). If a non-empty -wal exists
        but -shm does not (crash leftover), its content is ignored and a note is recorded."""
        try:
            with open(path, 'rb') as f:
                hdr = f.read(20)
        except OSError:
            return 'ro'
        is_wal = len(hdr) >= 20 and hdr[18] == 2 and hdr[19] == 2
        if not is_wal:
            return 'ro'
        wal, shm = path + '-wal', path + '-shm'
        if os.path.exists(wal) and os.path.exists(shm):
            return 'ro'
        if os.path.exists(wal) and os.path.getsize(wal) > 0:
            self.R['sections'].setdefault('notes', []).append(
                f'{os.path.basename(path)}: non-empty -wal without -shm; opened immutable, WAL contents ignored.')
        else:
            self.R['sections'].setdefault('notes', []).append(
                f'{os.path.basename(path)}: WAL database with no live -wal/-shm; opened immutable to avoid creating them.')
        return 'ro&immutable=1'

    def _progress(self):
        return 1 if (self.deadline is not None and time.monotonic() > self.deadline) else 0

    def connect(self, label, fname, tune=False):
        path = os.path.join(self.media_dir, fname)
        if not os.path.isfile(path):
            self.conns[label] = None
            return None
        try:
            conn = self.open_ro(path, tune)
            conn.execute('SELECT count(*) FROM sqlite_master').fetchone()
        except sqlite3.Error as e:
            self.R['sections'].setdefault('errors', []).append(f'{fname}: cannot open read-only: {e}')
            self.conns[label] = None
            return None
        self.conns[label] = conn
        return conn

    def q(self, conn, sql, params=(), timeout=None):
        """Run a query with the deadline armed; returns fetchall() or raises."""
        self.deadline = time.monotonic() + (timeout or self.timeout)
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            self.deadline = None

    def q1(self, conn, sql, params=(), default=None, timeout=None):
        try:
            rows = self.q(conn, sql, params, timeout)
            return rows[0][0] if rows else default
        except sqlite3.Error:
            return default

    def timed(self, conn, sql, params=(), timeout=None):
        """Run `sql` self.runs times, streaming results (not retained). Returns timings,
        row count and approx payload MB."""
        out = {'runs_s': [], 'rows': None, 'mb': None, 'error': None}
        for _ in range(self.runs):
            self.deadline = time.monotonic() + (timeout or self.timeout)
            t0 = time.perf_counter()
            try:
                cur = conn.execute(sql, params)
                n = 0
                nbytes = 0
                while True:
                    rows = cur.fetchmany(5000)
                    if not rows:
                        break
                    n += len(rows)
                    for r in rows:
                        for v in r:
                            nbytes += len(v) if isinstance(v, (bytes, str)) else 8
            except sqlite3.Error as e:
                msg = str(e)
                out['error'] = ('timeout (> %.0fs)' % (timeout or self.timeout)
                                if 'interrupt' in msg.lower() else
                                ('missing: ' + msg if 'no such' in msg else msg))
                break
            finally:
                self.deadline = None
            out['runs_s'].append(time.perf_counter() - t0)
            out['rows'] = n
            out['mb'] = round(nbytes / MB, 2)
        if out['runs_s']:
            out['min_s'] = round(min(out['runs_s']), 4)
            out['median_s'] = round(statistics.median(out['runs_s']), 4)
            out['runs_s'] = [round(x, 4) for x in out['runs_s']]
        return out

    def plan(self, conn, sql, params=()):
        try:
            rows = self.q(conn, 'EXPLAIN QUERY PLAN ' + sql, params, timeout=15)
        except sqlite3.Error as e:
            return {'lines': [], 'flags': [], 'error': ('missing: ' + str(e)) if 'no such' in str(e) else str(e)}
        lines = [r[3] for r in rows]
        flags = []
        for ln in lines:
            u = ln.upper()
            if u.startswith('SCAN') and 'CONSTANT ROW' not in u:
                flags.append('SCAN(index)' if 'USING' in u and 'INDEX' in u else 'SCAN')
            elif u.startswith('SEARCH'):
                flags.append('SEARCH')
            if 'TEMP B-TREE' in u:
                flags.append('TEMP B-TREE')
            if 'CORRELATED' in u:
                flags.append('CORRELATED SUBQUERY')
        return {'lines': lines, 'flags': sorted(set(flags)), 'error': None}

    def finding(self, score_s, title, evidence, fix, kind='perf'):
        """score_s: seconds-equivalent cost, used only to rank. Size findings are scored
        as the time to read that many bytes at 200 MB/s."""
        self.findings.append({'score': float(score_s), 'kind': kind, 'title': title,
                              'evidence': evidence, 'fix': fix})

    # -- 1. files & pragmas ------------------------------------------------------
    def section_files(self):
        sec = {}
        for fname in ('media.db', 'media.db-wal', 'media.db-shm', 'manual.db', 'manual.db-wal',
                      'manual.db-shm', 'error.db', 'error.db-wal'):
            p = os.path.join(self.media_dir, fname)
            sec[fname] = os.path.getsize(p) if os.path.isfile(p) else None
        self.R['sections']['files'] = sec
        pr = {}
        for label, conn in (('media.db', self.conns.get('media')), ('manual.db', self.conns.get('manual')),
                            ('error.db', self.conns.get('error'))):
            if conn is None:
                pr[label] = 'missing'
                continue
            d = {}
            for p in ('page_size', 'page_count', 'freelist_count', 'journal_mode', 'synchronous',
                      'cache_size', 'mmap_size', 'wal_autocheckpoint', 'auto_vacuum', 'user_version',
                      'encoding'):
                try:
                    d[p] = conn.execute(f'PRAGMA {p}').fetchone()[0]
                except (sqlite3.Error, TypeError):
                    d[p] = 'missing'
            try:
                d['wasted_pct'] = round(100.0 * d['freelist_count'] / d['page_count'], 2) if d['page_count'] else 0.0
                d['wasted_bytes'] = d['freelist_count'] * d['page_size']
            except TypeError:
                pass
            try:  # immutable opens report 'delete'; the file header (byte 18 == 2) is authoritative
                with open(os.path.join(self.media_dir, label), 'rb') as f:
                    hdr = f.read(20)
                d['journal_mode'] = 'wal' if hdr[18] == 2 else d['journal_mode']
            except OSError:
                pass
            d['sqlite_version'] = sqlite3.sqlite_version
            pr[label] = d
        pr['_note'] = ('synchronous/cache_size/mmap_size/temp_store are per-connection settings; '
                       'values shown are this audit connection (media.db tuned like the app: '
                       'synchronous=NORMAL, cache 128MB, mmap 4096MB by default via '
                       'MEDIA_DB_* env). journal_mode/page_size/auto_vacuum are persistent.')
        self.R['sections']['pragmas'] = pr

        m = pr.get('media.db')
        if isinstance(m, dict) and m.get('page_size') not in (None, 'missing'):
            wal = sec.get('media.db-wal') or 0
            if wal > 100 * MB:
                self.finding(wal / (200 * MB), 'media.db WAL is %.0f MB' % (wal / MB),
                             'WAL > 100MB: readers pay extra to consult it and it will not shrink until a '
                             'checkpoint can finish (a long-lived reader or constant writers block it).',
                             'Stop long-lived reader connections; run `PRAGMA wal_checkpoint(TRUNCATE)` in a '
                             'quiet moment; consider `PRAGMA journal_size_limit=134217728`; check for '
                             'connections that never close (see FD-leak history).', 'wal')
            if isinstance(m.get('wasted_bytes'), int) and m['wasted_bytes'] > 0 and \
                    (m['wasted_pct'] >= 10 or m['wasted_bytes'] > 1 * GB):
                self.finding(m['wasted_bytes'] / (200 * MB),
                             'media.db freelist is %.1f%% (%.2f GB reclaimable)' % (m['wasted_pct'], m['wasted_bytes'] / GB),
                             'freelist_count=%d pages x %d bytes' % (m['freelist_count'], m['page_size']),
                             'VACUUM (needs ~DB-size free disk, app stopped) or `PRAGMA auto_vacuum=INCREMENTAL` '
                             '+ VACUUM once, then `PRAGMA incremental_vacuum` periodically.', 'size')

    # -- 2. tables / indexes -----------------------------------------------------
    def table_sizes(self, conn, label, deep=True):
        ps = conn.execute('PRAGMA page_size').fetchone()[0]
        objs = conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master "
                            "WHERE name NOT LIKE 'sqlite_%' OR type='index'").fetchall()
        tables = [o[1] for o in objs if o[0] == 'table' and not o[1].startswith('sqlite_')]
        idx_of = {}
        for o in objs:
            if o[0] == 'index':
                idx_of.setdefault(o[2], []).append(o[1])
        stat = {}
        dbstat_ok = False
        if deep and not self.quick:
            try:
                rows = self.q(conn,
                              "SELECT name, SUM(pgsize), SUM(unused), SUM(payload), "
                              "SUM(CASE WHEN pagetype='leaf' THEN ncell ELSE 0 END), "
                              "SUM(pagetype='overflow'), SUM(pagetype='leaf'), COUNT(*) "
                              "FROM dbstat GROUP BY name", timeout=self.timeout * 3)
                for r in rows:
                    stat[r[0]] = dict(bytes=r[1], unused=r[2], payload=r[3], cells=r[4],
                                      overflow_pages=r[5], leaf_pages=r[6], pages=r[7])
                dbstat_ok = True
            except sqlite3.Error as e:
                stat = {}
                self.R['sections'].setdefault('notes', []).append(
                    f'{label}: dbstat unavailable/timeout ({e}); using row counts + blob lengths.')
        file_bytes = None
        total_pages = conn.execute('PRAGMA page_count').fetchone()[0]
        file_bytes = total_pages * ps
        out = {}
        for t in tables:
            e = {'rows': None, 'indexes': {}, 'flags': [], 'blob_cols': {}}
            e['rows'] = self.q1(conn, f'SELECT COUNT(*) FROM "{t}"', default=None)
            cols = conn.execute(f'PRAGMA table_info("{t}")').fetchall()
            for c in cols:
                if (c[2] or '').upper() == 'BLOB':
                    try:
                        avg = self.q(conn, f'SELECT AVG(length("{c[1]}")), MAX(length("{c[1]}")) FROM '
                                           f'(SELECT "{c[1]}" FROM "{t}" LIMIT 5000)', timeout=15)[0]
                        e['blob_cols'][c[1]] = {'avg_len': round(avg[0], 1) if avg[0] is not None else None,
                                                'max_len': avg[1]}
                    except sqlite3.Error:
                        pass
            s = stat.get(t)
            if s:
                e.update(bytes=s['bytes'], unused_bytes=s['unused'], overflow_pages=s['overflow_pages'],
                         leaf_pages=s['leaf_pages'], source='dbstat')
                e['unused_pct'] = round(100.0 * s['unused'] / s['bytes'], 1) if s['bytes'] else 0
                if e['rows']:
                    e['rows_per_leaf_page'] = round(e['rows'] / max(1, s['leaf_pages']), 2)
                    e['avg_row_bytes'] = round(s['bytes'] / e['rows'], 1)
            else:
                est = None
                if e['rows'] is not None:
                    blob = sum((b['avg_len'] or 0) for b in e['blob_cols'].values())
                    est = int(e['rows'] * (blob + 40))
                e.update(bytes=est, source='estimate (rows x (avg blob + 40))')
                if e['rows']:
                    e['avg_row_bytes'] = round(est / e['rows'], 1) if est else None
            e['pct_of_file'] = round(100.0 * e['bytes'] / file_bytes, 2) if e.get('bytes') and file_bytes else None
            # overflow / page-fit flag
            blob_avg = sum((b['avg_len'] or 0) for b in e['blob_cols'].values())
            if s and s['overflow_pages'] > 0:
                e['flags'].append('BLOB OVERFLOW: %d overflow pages' % s['overflow_pages'])
            elif blob_avg > ps - 100:
                e['flags'].append('BLOB OVERFLOW (estimated): avg blob %d B > page_size-100 (%d)' % (blob_avg, ps - 100))
            if e.get('avg_row_bytes') and e['avg_row_bytes'] > ps and 'BLOB OVERFLOW' not in ' '.join(e['flags']):
                e['flags'].append('row > page_size (avg %.0f B vs %d)' % (e['avg_row_bytes'], ps))
            if t in BLOB_FLAG_TABLES and e.get('rows_per_leaf_page') is not None and e['rows_per_leaf_page'] < 2 and blob_avg:
                e['flags'].append('<2 rows per leaf page: page_size %d wastes space for %.0f B rows' % (ps, blob_avg))
            for ix in idx_of.get(t, []):
                ixs = stat.get(ix)
                icols = [r[2] for r in conn.execute(f'PRAGMA index_info("{ix}")').fetchall()]
                e['indexes'][ix] = {'columns': icols, 'bytes': ixs['bytes'] if ixs else None,
                                    'pct_of_file': round(100.0 * ixs['bytes'] / file_bytes, 2) if ixs and file_bytes else None}
            # autoindexes (PK/UNIQUE) are not in sqlite_master with sql; list via index_list
            for r in conn.execute(f'PRAGMA index_list("{t}")').fetchall():
                if r[1] not in e['indexes']:
                    icols = [x[2] for x in conn.execute(f'PRAGMA index_info("{r[1]}")').fetchall()]
                    ixs = stat.get(r[1])
                    e['indexes'][r[1]] = {'columns': icols, 'bytes': ixs['bytes'] if ixs else None,
                                          'pct_of_file': round(100.0 * ixs['bytes'] / file_bytes, 2) if ixs and file_bytes else None}
            out[t] = e
        return {'file_bytes': file_bytes, 'page_size': ps, 'dbstat': dbstat_ok, 'tables': out,
                'views': [o[1] for o in objs if o[0] == 'view']}

    def section_tables(self):
        res = {}
        for label, key in (('media.db', 'media'), ('manual.db', 'manual'), ('error.db', 'error')):
            conn = self.conns.get(key)
            res[label] = self.table_sizes(conn, label) if conn is not None else 'missing'
        self.R['sections']['tables'] = res
        media = res['media.db']
        if not isinstance(media, dict):
            return
        T = media['tables']
        # float16 saving: half of the blob payload of float32 embedding tables
        saving = 0
        for t in ('embeddings', 'embeddings_siglip', 'embeddings_dino', 'tile_embeddings',
                  'body_embeddings', 'object_region_embeddings', 'place_embeddings'):
            e = T.get(t)
            if not e or not e.get('rows'):
                continue
            avg = sum((b['avg_len'] or 0) for b in e['blob_cols'].values())
            saving += e['rows'] * avg / 2
        if saving > 0.25 * GB:
            self.finding(saving / (200 * MB), 'float32 embeddings could be float16: ~%.2f GB smaller' % (saving / GB),
                         'sum over embedding tables of rows x avg blob length / 2',
                         'Store float16 (halves disk, page cache and the matrix-load read volume); '
                         'cast to float32 once when building the in-RAM matrix.', 'size')
        for t, e in T.items():
            if t in BLOB_FLAG_TABLES and e['flags'] and e.get('unused_bytes') and e['unused_bytes'] > 0.5 * GB:
                self.finding(e['unused_bytes'] / (200 * MB),
                             f'{t}: {e["unused_bytes"] / GB:.2f} GB unused space inside pages ({e.get("unused_pct")}%)',
                             '; '.join(e['flags']),
                             'Page-size/blob layout waste: rebuild with a larger page_size (`PRAGMA page_size=16384; VACUUM`) '
                             'or move vectors into a side table / flat memory-mapped file so rows are not spilled to overflow chains.',
                             'size')
        # tables the code never references
        src = self._source_text()
        if src is not None:
            unref = []
            for t, e in T.items():
                refs = sum(1 for ln in src if re.search(r'\b%s\b' % re.escape(t), ln)
                           and not re.search(r'CREATE\s+(TABLE|INDEX|UNIQUE)', ln, re.I))
                e['code_references'] = refs
                if refs == 0:
                    unref.append(t)
                    if (e.get('bytes') or 0) > 100 * MB:
                        self.finding((e['bytes'] or 0) / (200 * MB),
                                     f'{t}: {(e["bytes"] or 0) / GB:.2f} GB, never referenced by the code',
                                     f'rows={e["rows"]}', f'Candidate to DROP TABLE {t} (verify first; then VACUUM).', 'size')
            media['unreferenced_tables'] = unref

    def _source_text(self):
        pkg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'media_manager')
        if not os.path.isdir(pkg):
            return None
        lines = []
        for fn in os.listdir(pkg):
            if fn.endswith('.py'):
                try:
                    with open(os.path.join(pkg, fn), encoding='utf-8', errors='replace') as f:
                        lines.extend(f.read().splitlines())
                except OSError:
                    pass
        return lines

    # -- 3. embedding health -----------------------------------------------------
    def section_embeddings(self):
        conn = self.conns.get('media')
        health = {}
        for t in EMBEDDING_TABLES:
            h = {}
            try:
                cols = {r[1] for r in conn.execute(f'PRAGMA table_info("{t}")').fetchall()}
            except sqlite3.Error:
                cols = set()
            if not cols:
                health[t] = 'missing'
                continue
            ecol = 'embedding' if 'embedding' in cols else None
            if ecol is None:
                health[t] = 'missing (no embedding column)'
                continue
            try:
                h['rows'] = self.q1(conn, f'SELECT COUNT(*) FROM "{t}"')
                if 'model' in cols:
                    rows = self.q(conn, f'SELECT model, length({ecol})/4 AS dim, COUNT(*) FROM "{t}" '
                                        f'GROUP BY model, dim ORDER BY 3 DESC LIMIT 30')
                    h['model_dim_counts'] = [{'model': r[0], 'dim': r[1], 'rows': r[2]} for r in rows]
                else:
                    rows = self.q(conn, f'SELECT length({ecol})/4 AS dim, COUNT(*) FROM "{t}" GROUP BY dim '
                                        f'ORDER BY 2 DESC LIMIT 30')
                    h['model_dim_counts'] = [{'model': None, 'dim': r[0], 'rows': r[1]} for r in rows]
                h['distinct_dims'] = sorted({x['dim'] for x in h['model_dim_counts'] if x['dim'] is not None})
                h['bytes_not_multiple_of_4'] = self.q1(conn, f'SELECT COUNT(*) FROM "{t}" WHERE length({ecol}) % 4 != 0')
                anomalies = {}
                anomalies['null_or_empty_embedding'] = self.q1(
                    conn, f'SELECT COUNT(*) FROM "{t}" WHERE {ecol} IS NULL OR length({ecol}) = 0')
                if 'model' in cols:
                    anomalies['null_model'] = self.q1(conn, f'SELECT COUNT(*) FROM "{t}" WHERE model IS NULL')
                if 'file_id' in cols:
                    anomalies['orphan_file_id'] = self.q1(
                        conn, f'SELECT COUNT(*) FROM "{t}" e WHERE NOT EXISTS (SELECT 1 FROM files f WHERE f.id = e.file_id)')
                if 'frame_index' in cols and 'file_id' in cols:
                    fr = self.q(conn, f'SELECT frame_index, COUNT(*) FROM "{t}" GROUP BY frame_index ORDER BY 2 DESC LIMIT 12')
                    h['rows_per_frame_index_top'] = [{'frame_index': r[0], 'rows': r[1]} for r in fr]
                    h['distinct_frame_index'] = self.q1(conn, f'SELECT COUNT(DISTINCT frame_index) FROM "{t}"')
                    if 'id' in cols:
                        anomalies['duplicate_file_frame'] = self.q1(
                            conn, f'SELECT COUNT(*) FROM (SELECT 1 FROM "{t}" GROUP BY file_id, frame_index HAVING COUNT(*) > 1)')
                    h['files_with_rows'] = self.q1(conn, f'SELECT COUNT(DISTINCT file_id) FROM "{t}"')
                elif 'tile_index' in cols:
                    anomalies['duplicate_file_tile'] = self.q1(
                        conn, f'SELECT COUNT(*) FROM (SELECT 1 FROM "{t}" GROUP BY file_id, tile_index HAVING COUNT(*) > 1)')
                h['anomalies'] = anomalies
            except sqlite3.Error as e:
                h['error'] = str(e)
            health[t] = h
        # files lacking a primary embedding per space
        missing = {}
        total_files = self.q1(conn, 'SELECT COUNT(*) FROM files')
        nonhidden = self.q1(conn, 'SELECT COUNT(*) FROM files WHERE hidden = 0')
        missing['_files_total'] = total_files
        missing['_files_non_hidden'] = nonhidden
        missing['_files_without_any_path'] = self.q1(
            conn, 'SELECT COUNT(*) FROM files f WHERE NOT EXISTS (SELECT 1 FROM file_paths p WHERE p.file_id = f.id)')
        for sp, t in SPACES.items():
            if not isinstance(health.get(t), dict):
                missing[sp] = 'missing'
                continue
            missing[sp] = self.q1(conn, f'SELECT COUNT(*) FROM files f WHERE NOT EXISTS '
                                        f'(SELECT 1 FROM "{t}" e WHERE e.file_id = f.id AND e.frame_index = 0)',
                                  default='missing', timeout=self.timeout * 2)
        self.R['sections']['embedding_health'] = {'tables': health, 'files_without_primary_embedding': missing}
        for t, h in health.items():
            if isinstance(h, dict) and len(h.get('distinct_dims') or []) > 1 and t in SPACES.values():
                self.finding(30, f'{t}: mixed dimensions {h["distinct_dims"]}',
                             'Mixed-dim rows are dropped when the matrix is stacked.',
                             'Re-embed or delete rows from the stale model.', 'health')

    # -- 5/6. hot queries --------------------------------------------------------
    def build_context(self):
        conn = self.conns['media']
        ctx = {}
        ctx['space_counts'] = {sp: self.q1(conn, f'SELECT COUNT(*) FROM "{t}" WHERE frame_index = 0', default=0)
                               for sp, t in SPACES.items()}
        ctx['models'] = {}
        for sp, t in SPACES.items():
            try:
                ctx['models'][sp] = self.q1(conn, f'SELECT model FROM "{t}" GROUP BY model ORDER BY COUNT(*) DESC LIMIT 1',
                                            default='unknown-model')
            except Exception:
                ctx['models'][sp] = 'unknown-model'
        p = self.q1(conn, 'SELECT path FROM file_paths WHERE id >= (SELECT MAX(id)/2 FROM file_paths) LIMIT 1')
        ctx['sample_path'] = p
        if p:
            parts = p.split('/')
            # a folder two levels up (a realistic /files browse prefix), else the parent
            ctx['glob_folder'] = '/'.join(parts[:-2]) if len(parts) > 3 else '/'.join(parts[:-1])
        else:
            ctx['glob_folder'] = None
        try:
            ctx['checksums'] = [r[0] for r in self.q(conn, 'SELECT checksum FROM files WHERE id >= (SELECT MAX(id)/2 FROM files) LIMIT 500')]
        except sqlite3.Error:
            ctx['checksums'] = []
        ctx['total_files'] = self.q1(conn, 'SELECT COUNT(*) FROM files', default=0)
        return ctx

    def hot_queries(self, ctx):
        S = []

        def add(name, sql, params=(), slow=False, expect_scan=False, fix='', group='misc', note=''):
            S.append(dict(name=name, sql=sql, params=params, slow=slow, expect_scan=expect_scan, fix=fix,
                          group=group, note=note))
        VIEW_FIX = ('Replace the per-row correlated subquery: add covering index file_paths(file_id, last_seen_at DESC) '
                    'and/or materialise primary_path_id on files (maintained by upsert_file_path); see section 6 '
                    'for the measured alternatives.')
        add('view_count', 'SELECT COUNT(*) FROM files_with_path', group='view', fix=VIEW_FIX, expect_scan=True)
        add('view_id_path', 'SELECT id, path FROM files_with_path', slow=True, group='view', fix=VIEW_FIX, expect_scan=True)
        add('files_count_baseline', 'SELECT COUNT(*) FROM files', group='view', expect_scan=True)
        add('files_nonhidden_count', 'SELECT COUNT(*) FROM files WHERE hidden = 0', group='counts', expect_scan=True,
            fix='Partial index files(id) WHERE hidden=0 or maintain a counter row.')
        for sp, t in SPACES.items():
            n = ctx['space_counts'].get(sp) or 0
            add(f'matrix_{sp}_ids', f'SELECT e.file_id, f.checksum FROM "{t}" e JOIN files_with_path f ON f.id = e.file_id '
                                    f'WHERE e.frame_index = 0', slow=True, group='matrix', expect_scan=True, fix=VIEW_FIX +
                ' The matrix join resolves a path per embedding row only to read the checksum: join `files` directly '
                '(checksum lives there) and skip files_with_path.')
            add(f'matrix_{sp}_ids_files_only', f'SELECT e.file_id, f.checksum FROM "{t}" e JOIN files f ON f.id = e.file_id '
                                               f'WHERE e.frame_index = 0', slow=True, group='matrix', expect_scan=True,
                note='alternative: same result without the view')
            S[-1]['alt_of'] = f'matrix_{sp}_ids'
            lim = 20000
            S.append(dict(name=f'matrix_{sp}_blobs_first{lim}', sql=f'SELECT e.file_id, f.checksum, e.embedding FROM "{t}" e '
                                                                  f'JOIN files_with_path f ON f.id = e.file_id WHERE e.frame_index = 0 LIMIT {lim}',
                          params=(), slow=True, expect_scan=True, group='matrix', note='', extrapolate_to=n,
                          fix='Vector load volume: float16 storage, larger page_size, or a flat .npy cache keyed by '
                              'a table version to avoid re-reading blobs from SQLite at startup.'))
            S.append(dict(name=f'matrix_{sp}_blobs_plain_first{lim}', sql=f'SELECT file_id, embedding FROM "{t}" '
                                                                         f'WHERE frame_index = 0 AND embedding IS NOT NULL ORDER BY file_id LIMIT {lim}',
                          params=(), slow=True, expect_scan=True, group='matrix', extrapolate_to=n,
                          fix='', note='blob-only read (no join) = lower bound for the matrix load'))
            m = ctx['models'].get(sp) or 'x'
            add(f'needing_{sp}_count', f'SELECT COUNT(*) FROM (SELECT f.id, f.path FROM files_with_path f LEFT JOIN "{t}" e '
                                       f'ON e.file_id = f.id AND e.frame_index = 0 WHERE e.file_id IS NULL OR e.model != ?)',
                (m,), slow=True, group='needing', expect_scan=True, fix=VIEW_FIX +
                ' Also: the OR (e.file_id IS NULL OR e.model != ?) defeats anti-join planning; split into NOT EXISTS + model query on files (not the view).')
            add(f'needing_{sp}_limit100', f'SELECT f.id, f.path FROM files_with_path f LEFT JOIN "{t}" e '
                                         f'ON e.file_id = f.id AND e.frame_index = 0 WHERE e.file_id IS NULL OR e.model != ? LIMIT 100',
                (m,), group='needing', expect_scan=True)
        gf = ctx.get('glob_folder')
        if gf:
            esc = gf.translate(str.maketrans({'*': '[*]', '?': '[?]', '[': '[[]', ']': '[]]'}))
            add('path_glob_bound', 'SELECT f.id, fp.path, f.checksum FROM file_paths fp JOIN files f ON f.id = fp.file_id '
                                   'WHERE (fp.path = ? OR fp.path GLOB ?) AND f.hidden = 0', (gf, esc + '/*'), group='path',
                fix='If SCAN: use range predicate path >= ? AND path < ? (prefix, prefix with last char+1) instead of GLOB.')
            lo, hi = gf + '/', gf + '0'  # '/' + 1 == '0'
            add('path_range_alt', 'SELECT f.id, fp.path, f.checksum FROM file_paths fp JOIN files f ON f.id = fp.file_id '
                                  'WHERE (fp.path = ? OR (fp.path >= ? AND fp.path < ?)) AND f.hidden = 0', (gf, lo, hi),
                group='path', note='alternative formulation using a half-open range')
            S[-1]['alt_of'] = 'path_glob_bound'
        cs = ctx.get('checksums') or []
        if cs:
            ph = ','.join('?' for _ in cs)
            add('checksums_500_view', f'SELECT * FROM files_with_path WHERE checksum IN ({ph})', tuple(cs), group='lookup', fix=VIEW_FIX)
            add('checksums_500_files_join', f'SELECT f.*, fp.path FROM files f JOIN file_paths fp ON fp.file_id = f.id WHERE f.checksum IN ({ph})',
                tuple(cs), group='lookup', note='alternative, uses file_paths directly (all paths, not only primary)')
            S[-1]['alt_of'] = 'checksums_500_view'
        add('random_view_50', 'SELECT id, path, checksum FROM files_with_path ORDER BY RANDOM() LIMIT 50', slow=True,
            group='random', expect_scan=True, fix='Sample ids on `files` (ORDER BY RANDOM() LIMIT n or id >= random) and only then resolve paths for those 50 rows.')
        add('random_files_hidden0_50', 'SELECT checksum FROM files WHERE hidden = 0 ORDER BY RANDOM() LIMIT 50', group='random',
            expect_scan=True, fix='Pick random rowids (random id range probe) instead of sorting the whole table.')
        add('random_faceless_50', "SELECT id, checksum FROM files f WHERE f.hidden = 0 AND f.noface = 0 AND NOT EXISTS "
                                  "(SELECT 1 FROM faces fa WHERE fa.file_id = f.id AND (fa.identity IS NULL OR fa.identity != '__indexed__')) "
                                  "ORDER BY RANDOM() LIMIT 50", slow=True, group='random', expect_scan=True,
            fix='Probe random ids and test the NOT EXISTS only for candidates, rather than ranking the full table.')
        add('least_viewed_400', 'SELECT id, checksum FROM files WHERE hidden = 0 AND noface = 0 ORDER BY view_count ASC, RANDOM() LIMIT 400',
            slow=True, group='random', expect_scan=True, fix='Index files(view_count) (partial WHERE hidden=0 AND noface=0) so ORDER BY view_count needs no full sort.')
        add('random_by_ext_50', "SELECT id, path, checksum FROM files_with_path WHERE hidden = 0 AND (lower(path) LIKE '%' || ?) "
                                "ORDER BY RANDOM() LIMIT 50", ('.jpg',), slow=True, group='random', expect_scan=True, fix=VIEW_FIX +
            ' lower(path) LIKE %ext cannot use an index: store an `ext`/is_video column on files.')
        add('recent_by_ext_50', "SELECT id, path, checksum FROM files_with_path WHERE hidden = 0 AND (lower(path) LIKE '%' || ?) "
                                "ORDER BY first_seen DESC, id DESC LIMIT 50", ('.jpg',), group='random', expect_scan=True)
        add('count_by_ext', "SELECT COUNT(*) FROM files_with_path WHERE hidden = 0 AND (lower(path) LIKE '%' || ?)", ('.jpg',),
            slow=True, group='counts', expect_scan=True, fix=VIEW_FIX)
        add('count_embeddings_frame0', 'SELECT COUNT(*) FROM embeddings WHERE frame_index = 0', group='counts', expect_scan=True)
        add('count_faces_unidentified', 'SELECT COUNT(*) FROM faces WHERE identity IS NULL', group='counts', expect_scan=True)
        add('count_faces_unsuggested', "SELECT COUNT(*) FROM faces WHERE identity IS NULL AND suggested_score IS NULL AND embedding != x''",
            group='counts', expect_scan=True, slow=True)
        add('count_faces_suggested_pool', 'SELECT COUNT(*) FROM faces WHERE identity IS NULL AND handled = 0 AND suggested_score >= 0.5',
            group='counts', expect_scan=True)
        add('count_faces_unnormalized', "SELECT COUNT(*) FROM faces WHERE norm_version < 1 AND bbox != '[]'", group='counts', expect_scan=True)
        add('count_detected_files', 'SELECT COUNT(DISTINCT file_id) FROM detections WHERE frame_index IS NULL', group='counts', expect_scan=True)
        add('count_detected_classes', "SELECT COUNT(DISTINCT class_name) FROM detections WHERE frame_index IS NULL AND class_name != '__indexed__'",
            group='counts', expect_scan=True)
        add('count_phashed', 'SELECT COUNT(DISTINCT file_id) FROM phashes', group='counts', expect_scan=True)
        add('tags_group_by', 'SELECT tag, COUNT(*) AS cnt FROM tags GROUP BY tag ORDER BY cnt DESC', group='counts', expect_scan=True)
        add('faces_identity_group', 'SELECT identity, COUNT(*) FROM faces GROUP BY identity', group='counts', expect_scan=True)
        return S

    def manual_queries(self):
        return [
            ('manual_file_tags_count', 'SELECT COUNT(*) FROM file_tags', ()),
            ('manual_faces_identity_group', 'SELECT identity, COUNT(*) FROM faces GROUP BY identity', ()),
            ('manual_faces_count', 'SELECT COUNT(*) FROM faces', ()),
            ('manual_tag_label_distinct', "SELECT label, COUNT(DISTINCT checksum) FROM file_tags_with_label GROUP BY label", ()),
            ('manual_favorites_count', 'SELECT COUNT(*) FROM file_favorites', ()),
        ]

    def section_queries(self):
        conn = self.conns['media']
        ctx = self.build_context()
        self.R['sections']['context'] = {k: (v if k != 'checksums' else f'{len(v)} sampled checksums') for k, v in ctx.items()}
        res = {}
        for s in self.hot_queries(ctx):
            r = {'sql': ' '.join(s['sql'].split()), 'group': s['group'], 'note': s.get('note', '')}
            if s.get('alt_of'):
                r['alt_of'] = s['alt_of']
            pl = self.plan(conn, s['sql'], s['params'])
            r['plan'] = pl['lines']
            r['plan_flags'] = pl['flags']
            if pl['error']:
                r['status'] = pl['error']
                res[s['name']] = r
                continue
            if self.quick and s['slow']:
                r['status'] = 'skipped (--quick)'
                res[s['name']] = r
                continue
            t = self.timed(conn, s['sql'], s['params'])
            r.update(t)
            r['status'] = t['error'] or 'ok'
            n_target = s.get('extrapolate_to')
            if n_target and t.get('rows') and t.get('median_s') is not None:
                scale = n_target / t['rows']
                r['extrapolated_rows'] = n_target
                r['extrapolated_median_s'] = round(t['median_s'] * scale, 2)
                r['extrapolated_mb'] = round(t['mb'] * scale, 1)
                r['extrapolation_note'] = 'linear from first %d rows; cold-cache runs would be slower' % t['rows']
            r['fix'] = s.get('fix', '')
            r['expect_scan'] = s['expect_scan']
            res[s['name']] = r
        self.R['sections']['hot_queries'] = res
        # manual.db
        mres = {}
        mc = self.conns.get('manual')
        if mc is None:
            mres = 'missing'
        else:
            for name, sql, params in self.manual_queries():
                r = {'sql': sql}
                pl = self.plan(mc, sql, params)
                r['plan'], r['plan_flags'] = pl['lines'], pl['flags']
                if pl['error']:
                    r['status'] = pl['error']
                else:
                    r.update(self.timed(mc, sql, params))
                    r['status'] = r.get('error') or 'ok'
                mres[name] = r
        self.R['sections']['manual_queries'] = mres
        self._query_findings(res)

    def _query_findings(self, res):
        for name, r in res.items():
            if r.get('status') != 'ok':
                continue
            cost = r.get('extrapolated_median_s') or r.get('median_s') or 0
            if r.get('group') == 'matrix' and name.endswith('_ids_files_only'):
                continue
            if cost < 0.5:
                continue
            flags = ','.join(r.get('plan_flags') or [])
            ev = f'median {cost:.2f}s' + (f' (~{r.get("extrapolated_mb")} MB extrapolated)' if r.get('extrapolated_mb') else
                                          (f' for {r["rows"]} rows / {r["mb"]} MB' if r.get('rows') is not None else ''))
            ev += f'; plan flags: {flags or "n/a"}'
            alt = None
            for n2, r2 in res.items():
                if r2.get('alt_of') == name and r2.get('status') == 'ok' and r2.get('median_s') is not None \
                        and r.get('median_s') and r2['median_s'] < r['median_s'] * 0.8:
                    alt = f'alternative `{n2}` measured {r2["median_s"]:.2f}s vs {r["median_s"]:.2f}s'
            fix = r.get('fix') or 'Review the plan above.'
            if alt:
                fix += ' ' + alt + '.'
            self.finding(cost, f'query `{name}` is slow', ev, fix)

    # -- 4. index audit ----------------------------------------------------------
    # (table, columns, used-by, kind). Derived by reading database.py; every miss is only
    # a 'candidate' - confirm against the plan flags in the hot-query section.
    ACCESS = [
        ('file_paths', ('file_id', 'last_seen_at'), 'files_with_path view: correlated subquery WHERE fp2.file_id=f.id ORDER BY last_seen_at DESC LIMIT 1 (runs per files row)'),
        ('file_paths', ('file_id',), 'view join / get_paths_for_file / exact-duplicate GROUP BY file_id'),
        ('file_paths', ('path',), 'path = ?, path GLOB prefix (find_files_under_folder), upsert_file_path'),
        ('files', ('checksum',), 'get_files_by_checksums / get_file_by_checksum / upsert'),
        ('files', ('hidden',), 'count_files, get_random_file_checksums, list/count_files_by_ext: WHERE hidden = 0 (low selectivity: partial-index candidate only)'),
        ('files', ('view_count',), 'get_least_viewed_files ORDER BY view_count ASC'),
        ('files', ('first_seen', 'id'), 'get_recent_files / gallery sort added'),
        ('files', ('taken_at',), 'gallery sort date'),
        ('files', ('city_id',), 'files WHERE city_id = ? / JOIN cities c ON f.city_id = c.id'),
        ('files', ('modified_time',), 'gallery sort modified (note: modified_time lives on file_paths, not files, in the new schema)'),
        ('embeddings', ('file_id', 'frame_index'), 'PK; get_embeddings_for_ids, matrix join'),
        ('embeddings', ('frame_index',), 'WHERE frame_index = 0 count; PK covers it only via full covering-index scan (partial index WHERE frame_index=0 candidate)'),
        ('embeddings_siglip', ('file_id', 'frame_index'), 'PK'),
        ('embeddings_dino', ('file_id', 'frame_index'), 'PK'),
        ('faces', ('file_id',), 'faces per file'),
        ('faces', ('identity',), 'GROUP BY identity / identity IS NULL counts'),
        ('faces', ('identity', 'handled', 'suggested_score'), 'count_suggested_unidentified_faces: identity IS NULL AND handled=0 AND suggested_score >= ?'),
        ('faces', ('norm_version',), 'count_unnormalized_faces'),
        ('detections', ('file_id', 'frame_index'), 'per-file detections'),
        ('detections', ('class_name',), 'class filters / COUNT(DISTINCT class_name)'),
        ('detections', ('frame_index',), 'WHERE frame_index IS NULL counts (index (file_id,frame_index) can only be scanned)'),
        ('tags', ('tag',), 'get_files_by_tag, list_all_tags GROUP BY'),
        ('tags', ('file_id', 'tag'), 'UNIQUE(file_id, tag) get_tags'),
        ('phashes', ('file_id',), 'count_phashed / joins'),
        ('tile_embeddings', ('file_id',), 'per-file tiles'),
        ('object_region_embeddings', ('file_id',), 'per-file regions'),
        ('body_embeddings', ('file_id', 'frame_index'), 'per-file bodies'),
        ('dup_group_members', ('file_id',), 'file -> group lookups (PK is group_id,file_id)'),
        ('cities', ('lat',), 'nearest_city'), ('cities', ('lon',), 'nearest_city'),
        ('file_tags', ('checksum',), 'manual.db: tags for a file'), ('file_tags', ('tag_id',), 'manual.db'),
        ('faces', ('checksum',), 'manual.db: faces for a file'),
        ('trash', ('checksum',), 'manual.db: trash membership'),
    ]

    @staticmethod
    def _has_prefix(indexes, cols):
        for ix in indexes.values():
            ic = [c for c in ix['columns'] if c is not None]
            if tuple(ic[:len(cols)]) == tuple(cols):
                return ix
        return None

    def section_indexes(self):
        tabs = self.R['sections'].get('tables', {})
        out = {'indexes': {}, 'access_patterns': []}
        for label in ('media.db', 'manual.db'):
            t = tabs.get(label)
            if not isinstance(t, dict):
                continue
            out['indexes'][label] = {name: {k: {'columns': v['columns'], 'bytes': v['bytes']} for k, v in e['indexes'].items()}
                                     for name, e in t['tables'].items()}
        for table, cols, why in self.ACCESS:
            found = 'missing table'
            status = 'missing table'
            for label in ('media.db', 'manual.db'):
                t = tabs.get(label)
                if not isinstance(t, dict) or table not in t['tables']:
                    continue
                e = t['tables'][table]
                tcols = {r[1] for r in self.conns['media' if label == 'media.db' else 'manual'].execute(f'PRAGMA table_info("{table}")')}
                if not set(cols) <= tcols:
                    status, found = 'missing column', None
                    continue
                ix = self._has_prefix(e['indexes'], cols)
                # rowid alias / INTEGER PRIMARY KEY covers ('id',) implicitly
                if ix is None and cols == ('id',):
                    ix = {'columns': ['id']}
                status = 'indexed' if ix else 'NO INDEX (candidate)'
                found = (next(k for k, v in e['indexes'].items() if v is ix) if ix and ix in e['indexes'].values() else None)
                out['access_patterns'].append({'db': label, 'table': table, 'columns': list(cols), 'used_by': why,
                                               'status': status, 'index': found})
                break
            else:
                out['access_patterns'].append({'db': None, 'table': table, 'columns': list(cols), 'used_by': why,
                                               'status': status, 'index': None})
        self.R['sections']['index_audit'] = out

    # -- 6. view alternatives ----------------------------------------------------
    def section_view_alternatives(self):
        conn = self.conns['media']
        alts = [
            ('A_view_current', 'SELECT id, path FROM files_with_path', 'current correlated-subquery view'),
            ('B_window_row_number',
             'SELECT id, path FROM (SELECT f.id AS id, fp.path AS path, ROW_NUMBER() OVER '
             '(PARTITION BY fp.file_id ORDER BY fp.last_seen_at DESC, fp.id DESC) AS rn '
             'FROM file_paths fp JOIN files f ON f.id = fp.file_id) WHERE rn = 1', 'window function'),
            ('C_max_join',
             'SELECT f.id, fp.path FROM files f JOIN (SELECT file_id, MAX(last_seen_at) AS m FROM file_paths GROUP BY file_id) mx '
             'ON mx.file_id = f.id JOIN file_paths fp ON fp.file_id = mx.file_id AND fp.last_seen_at = mx.m',
             'MAX(last_seen_at) join (may return >1 row per file on ties)'),
            ('D_bare_join_lower_bound', 'SELECT f.id, fp.path FROM files f JOIN file_paths fp ON fp.file_id = f.id',
             'NOT equivalent (all paths); pure join cost = lower bound'),
            ('E_file_paths_scan', 'SELECT COUNT(*) FROM file_paths', 'baseline: table scan of file_paths'),
        ]
        res = {}
        for name, sql, desc in alts:
            r = {'sql': sql, 'description': desc}
            pl = self.plan(conn, sql)
            r['plan'], r['plan_flags'] = pl['lines'], pl['flags']
            if pl['error']:
                r['status'] = pl['error']
            elif self.quick:
                r['status'] = 'skipped (--quick)'
            else:
                r.update(self.timed(conn, sql))
                r['status'] = r.get('error') or 'ok'
            res[name] = r
        self.R['sections']['view_alternatives'] = res
        a = res['A_view_current']
        if a.get('median_s'):
            best = min(((n, r) for n, r in res.items() if n[0] in 'BC' and r.get('median_s') is not None),
                       key=lambda x: x[1]['median_s'], default=None)
            if best and best[1]['median_s'] < a['median_s'] * 0.8:
                self.finding(a['median_s'] - best[1]['median_s'],
                             'files_with_path view is %.1fx slower than %s' % (a['median_s'] / best[1]['median_s'], best[0]),
                             f'view {a["median_s"]:.2f}s vs {best[0]} {best[1]["median_s"]:.2f}s for full id,path load (rows: {a["rows"]} vs {best[1]["rows"]})',
                             'Add covering index file_paths(file_id, last_seen_at DESC) and/or denormalise the primary path id onto files.')

    # -- findings ----------------------------------------------------------------
    def finish(self):
        # index-based finding
        ia = self.R['sections'].get('index_audit', {})
        for ap in ia.get('access_patterns', []):
            if ap['status'].startswith('NO INDEX') and ap['table'] == 'file_paths' and ap['columns'] == ['file_id', 'last_seen_at']:
                view = (self.R['sections'].get('hot_queries', {}).get('view_id_path') or {})
                cost = view.get('median_s') or 0
                self.finding(cost + 0.01, 'no index on file_paths(file_id, last_seen_at)',
                             'The files_with_path view runs one correlated subquery per file; the existing index on '
                             'file_paths(file_id) must fetch every row of that file and sort by last_seen_at.',
                             'CREATE INDEX idx_file_paths_file_seen ON file_paths(file_id, last_seen_at DESC) '
                             '(the index is covering for the subquery; measure with section 6 queries).', 'index')
        # redundant indexes: identical column list to another index (e.g. explicit index on a UNIQUE column)
        for label, t in self.R['sections'].get('tables', {}).items():
            if not isinstance(t, dict):
                continue
            for tn, e in t['tables'].items():
                seen = {}
                for ix, iv in e['indexes'].items():
                    key = tuple(iv['columns'])
                    if key in seen:
                        b = iv['bytes'] or 0
                        if b > 50 * MB:
                            self.finding(b / (200 * MB), f'{label}: index {ix} duplicates {seen[key]} on {tn}{list(key)}',
                                         f'{_b(b)} of redundant index (also slows every write)',
                                         f'DROP INDEX {ix} (keep the UNIQUE/autoindex or the explicit one, not both).', 'size')
                    else:
                        seen[key] = ix
        self.findings.sort(key=lambda f: -f['score'])
        self.R['findings'] = self.findings

    def run(self):
        media = self.connect('media', 'media.db', tune=True)
        self.connect('manual', 'manual.db')
        self.connect('error', 'error.db')
        self.section_files()
        if media is None:
            self.R['sections'].setdefault('errors', []).append('media.db missing or unreadable: only file-level info available')
            self.finish()
            return self.R
        self.section_tables()
        self.section_embeddings()
        self.section_queries()
        self.section_indexes()
        self.section_view_alternatives()
        self.finish()
        for c in self.conns.values():
            if c is not None:
                c.close()
        return self.R


# --------------------------------------------------------------------------- rendering
def _b(n):
    if n is None:
        return 'n/a'
    for unit, size in (('GB', GB), ('MB', MB), ('KB', 1024)):
        if abs(n) >= size:
            return f'{n / size:.2f} {unit}'
    return f'{n} B'


def render(R):
    L = []
    h = lambda s: L.extend(['', '=' * 78, s, '=' * 78])
    L.append('MEDIA MANAGER DB AUDIT (read-only)')
    L.append(f'data root : {R["data_root"]}')
    L.append(f'generated : {R["generated_at"]}   quick={R["quick"]}  timeout={R["timeout_s"]}s  runs={R["runs"]}')
    S = R['sections']
    for e in S.get('errors', []):
        L.append('ERROR: ' + e)
    h('1. FILES AND PRAGMAS')
    for k, v in S.get('files', {}).items():
        L.append(f'  {k:18s} {_b(v) if v is not None else "(absent)"}')
    for label, d in S.get('pragmas', {}).items():
        if label == '_note':
            continue
        L.append(f'  [{label}]')
        if not isinstance(d, dict):
            L.append(f'    {d}')
            continue
        L.append('    ' + ', '.join(f'{k}={d[k]}' for k in ('page_size', 'page_count', 'freelist_count', 'journal_mode',
                                                             'synchronous', 'cache_size', 'mmap_size', 'wal_autocheckpoint',
                                                             'auto_vacuum', 'user_version') if k in d))
        if 'wasted_pct' in d:
            L.append(f'    wasted space (freelist): {d["wasted_pct"]}% = {_b(d["wasted_bytes"])}')
    if '_note' in S.get('pragmas', {}):
        L.append('  note: ' + S['pragmas']['_note'])
    wal = S.get('files', {}).get('media.db-wal')
    h('7. WAL GROWTH RISK')
    if wal is None:
        L.append('  no media.db-wal present')
    else:
        L.append(f'  media.db-wal = {_b(wal)}' + ('   WARNING: > 100 MB' if wal > 100 * MB else '   ok (<= 100 MB)'))
    h('2. TABLES AND INDEXES (bytes on disk)')
    for label, t in S.get('tables', {}).items():
        L.append(f'  [{label}]')
        if not isinstance(t, dict):
            L.append(f'    {t}')
            continue
        L.append(f'    file {_b(t["file_bytes"])}, page_size {t["page_size"]}, dbstat={t["dbstat"]}, views={t["views"]}')
        L.append(f'    {"table":28s}{"rows":>10s}{"size":>12s}{"%file":>7s}{"unused%":>8s}{"rows/pg":>8s}  flags')
        for name, e in sorted(t['tables'].items(), key=lambda kv: -(kv[1].get('bytes') or 0)):
            L.append(f'    {name:28s}{str(e["rows"]):>10s}{_b(e.get("bytes")):>12s}{str(e.get("pct_of_file")):>7s}'
                     f'{str(e.get("unused_pct", "")):>8s}{str(e.get("rows_per_leaf_page", "")):>8s}  ' + '; '.join(e['flags']))
            for ix, iv in e['indexes'].items():
                L.append(f'      idx {ix:34s} ({",".join(str(c) for c in iv["columns"])})  {_b(iv["bytes"])}  {iv["pct_of_file"] if iv["pct_of_file"] is not None else ""}')
            for c, bv in e['blob_cols'].items():
                L.append(f'      blob {c}: avg {bv["avg_len"]} B, max {bv["max_len"]} B')
        if t.get('unreferenced_tables'):
            L.append('    tables never referenced by the code: ' + ', '.join(t['unreferenced_tables']))
    for n in S.get('notes', []):
        L.append('  note: ' + n)
    h('3. EMBEDDING TABLE HEALTH')
    eh = S.get('embedding_health', {})
    for t, d in eh.get('tables', {}).items():
        if not isinstance(d, dict):
            L.append(f'  {t}: {d}')
            continue
        L.append(f'  {t}: rows={d.get("rows")}  distinct dims={d.get("distinct_dims")}  files={d.get("files_with_rows")}  '
                 f'non-multiple-of-4={d.get("bytes_not_multiple_of_4")}')
        for x in d.get('model_dim_counts', [])[:8]:
            L.append(f'      model={x["model"]!r} dim={x["dim"]} rows={x["rows"]}')
        if 'rows_per_frame_index_top' in d:
            L.append(f'      distinct frame_index={d.get("distinct_frame_index")}; top: ' +
                     ', '.join(f'{x["frame_index"]}:{x["rows"]}' for x in d['rows_per_frame_index_top'][:6]))
        an = {k: v for k, v in (d.get('anomalies') or {}).items()}
        L.append(f'      anomalies: {an}')
        if d.get('error'):
            L.append(f'      error: {d["error"]}')
    L.append(f'  files without primary (frame 0) embedding: {eh.get("files_without_primary_embedding")}')
    h('4. INDEX AUDIT (access patterns found in database.py vs live indexes)')
    for ap in S.get('index_audit', {}).get('access_patterns', []):
        L.append(f'  {ap["status"]:22s} {ap["table"]}({",".join(ap["columns"])})' + (f'  via {ap["index"]}' if ap.get('index') else '') +
                 f'\n{"":24s}used by: {ap["used_by"]}')
    h('5. HOT QUERIES (EXPLAIN QUERY PLAN + wall time, min/median over runs)')

    def show_q(name, r):
        st = r.get('status', '?')
        head = f'  {name}: {st}'
        if 'min_s' in r:
            head += f'  min={r["min_s"]}s median={r["median_s"]}s rows={r["rows"]} payload={r["mb"]}MB'
        L.append(head)
        if 'extrapolated_median_s' in r:
            L.append(f'      EXTRAPOLATED full table: ~{r["extrapolated_median_s"]}s, ~{r["extrapolated_mb"]} MB ({r["extrapolation_note"]})')
        if r.get('plan_flags'):
            L.append(f'      flags: {", ".join(r["plan_flags"])}')
        for ln in r.get('plan', []):
            L.append(f'      | {ln}')
    for name, r in S.get('hot_queries', {}).items():
        show_q(name, r)
    L.append('  -- manual.db')
    mq = S.get('manual_queries')
    if isinstance(mq, dict):
        for name, r in mq.items():
            show_q(name, r)
    else:
        L.append(f'  {mq}')
    h('6. files_with_path ALTERNATIVE FORMULATIONS (id, path for every file; no DB changes)')
    for name, r in S.get('view_alternatives', {}).items():
        show_q(f'{name} [{r["description"]}]', r)
    h('8. FINDINGS (ranked by measured cost; size findings scored at 200 MB/s read-equivalent)')
    if not R.get('findings'):
        L.append('  none above thresholds')
    for i, f in enumerate(R.get('findings', []), 1):
        L.append(f'  {i}. [{f["kind"]}] {f["title"]}   (score {f["score"]:.1f}s)')
        L.append(f'       evidence: {f["evidence"]}')
        L.append(f'       fix     : {f["fix"]}')
    return '\n'.join(L) + '\n'


def run_audit(data_root, quick=False, timeout=60.0, runs=3):
    return Audit(data_root, quick=quick, timeout=timeout, runs=runs).run()


def main(argv=None):
    ap = argparse.ArgumentParser(description='Read-only audit of media.db / manual.db / error.db.')
    ap.add_argument('--data-root', default='.', help='dir containing .media/ (default: cwd)')
    ap.add_argument('--quick', action='store_true', help='skip slow whole-table timings and dbstat')
    ap.add_argument('--json', metavar='PATH', help='also write the full result as JSON')
    ap.add_argument('--timeout', type=float, default=60.0, help='per-query timeout seconds (default 60)')
    ap.add_argument('--runs', type=int, default=3, help='repetitions per timing (default 3)')
    a = ap.parse_args(argv)
    if not os.path.isdir(os.path.join(a.data_root, '.media')):
        print(f'error: {os.path.abspath(a.data_root)}/.media not found', file=sys.stderr)
        return 2
    R = run_audit(a.data_root, a.quick, a.timeout, a.runs)
    print(render(R))
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(R, f, indent=2, default=str)
    return 0


if __name__ == '__main__':
    sys.exit(main())

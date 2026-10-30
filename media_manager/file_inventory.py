"""
Disk inventory for the /files file manager: what is actually ON DISK under data_root,
independent of what's tracked in file_paths.

Two sources, both hash-free (stat only):
  * list_folder_live() — one os.scandir of the folder being viewed. Always fresh; it is
    what the folder's own file list is built from. It also refreshes that folder's
    direct-children rows in disk_files as a side effect.
  * disk_files — a media.db table filled by a background full walk (InventoryJob). Only
    used for the recursive per-subfolder rollups ("12 untracked under trips/"), which
    would otherwise need a walk of the whole subtree on every page view.

disk_files holds only files with a MEDIA_EXTENSIONS extension (ignored and system ones
included — the rollups count those as 'ignored'), never anything under .media/. Paths
are data_root-relative, '/'-separated, exactly like file_paths.path, so the two join
by plain string equality.

classify_path() applies the scan gate exactly as fast_scan does (extension first, then
the .media/captured_frames/generated system dirs, then .mediaignore via ignore.py), so
'media' here means "a `media add` would pick this file up".
"""
import os
import threading
import time

from .formats import IMAGE_EXTENSIONS, MEDIA_EXTENSIONS, VIDEO_EXTENSIONS

# Top-level dirs fast_scan always skips: .media (our own store), captured_frames
# (stills we register ourselves as hidden files), generated (app-made artifacts).
SYSTEM_DIRS = ('.media', 'captured_frames', 'generated')
# An inventory older than this is reported as stale (the UI offers a refresh).
STALE_AFTER = 3600
# Rows per INSERT batch / commit during the full walk — keeps write locks short so
# request threads writing media.db never wait long behind the walk.
_WALK_BATCH = 2000

_GLOB_ESCAPES = str.maketrans({'*': '[*]', '?': '[?]', '[': '[[]', ']': '[]]'})


def glob_escape(prefix):
    """Escape GLOB metacharacters so `f'{glob_escape(p)}/*'` matches the literal
    folder p (same escaping as database.browse_folder). A literal-prefix GLOB on a
    BINARY-collated indexed column is an index range scan."""
    return prefix.translate(_GLOB_ESCAPES)


def ensure_schema(conn):
    """Create the inventory tables (idempotent). disk_inventory_meta holds
    'refreshed_at' (last completed walk) and 'walk_started_at' (cross-process guard
    so several uvicorn workers don't all auto-start a walk)."""
    conn.execute('''
        CREATE TABLE IF NOT EXISTS disk_files (
            path TEXT PRIMARY KEY,
            size INTEGER,
            mtime REAL,
            seen_at INTEGER NOT NULL
        )
    ''')
    conn.execute('CREATE TABLE IF NOT EXISTS disk_inventory_meta (key TEXT PRIMARY KEY, value)')
    conn.commit()


def _meta_get(conn, key):
    row = conn.execute('SELECT value FROM disk_inventory_meta WHERE key = ?', (key,)).fetchone()
    return row[0] if row else None


def _meta_set(conn, key, value):
    conn.execute('INSERT INTO disk_inventory_meta (key, value) VALUES (?, ?) '
                 'ON CONFLICT(key) DO UPDATE SET value = excluded.value', (key, value))


def refreshed_at(conn):
    """Unix time the last full walk completed, or None if it never ran."""
    value = _meta_get(conn, 'refreshed_at')
    return int(value) if value is not None else None


def media_kind(path):
    """'image' | 'video' | None, by extension (formats.py)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return 'image'
    if ext in VIDEO_EXTENSIONS:
        return 'video'
    return None


def is_system_path(rel_path):
    """True for anything under a top-level SYSTEM_DIRS folder (or that folder itself)."""
    return rel_path.split('/', 1)[0] in SYSTEM_DIRS


def classify_path(rel_path, ignore_rules):
    """(category, kind) for a data_root-relative '/'-path, in fast_scan's gate order:
      'other'   — not a MEDIA_EXTENSIONS file (kind None)
      'system'  — under .media/, captured_frames/ or generated/
      'ignored' — matched by .mediaignore (ignore_rules: an ignore.IgnoreRules)
      'media'   — a scan would pick it up
    kind is 'image' | 'video' for the last three."""
    kind = media_kind(rel_path)
    if kind is None:
        return 'other', None
    if is_system_path(rel_path):
        return 'system', kind
    if ignore_rules.is_ignored(rel_path):
        return 'ignored', kind
    return 'media', kind


def _join(rel_folder, name):
    return f'{rel_folder}/{name}' if rel_folder else name


def list_folder_live(data_root, rel_folder, conn=None):
    """One os.scandir of data_root/rel_folder → (dirs, files):
      dirs:  [name, ...]  (a top-level .media is left out)
      files: [{'name', 'size', 'mtime'}, ...]
    Symlinks are skipped, like the scan's `find -type f` (which doesn't follow them).
    A missing/unreadable folder lists as empty. With `conn`, also syncs this folder's
    direct-children rows in disk_files (upsert changed media files, drop vanished
    ones) — only writing when something differs, so a plain view takes no write lock."""
    rel_folder = rel_folder.strip('/')
    abs_dir = os.path.join(data_root, rel_folder) if rel_folder else data_root
    dirs, files = [], []
    try:
        with os.scandir(abs_dir) as it:
            for entry in it:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if not (not rel_folder and entry.name == '.media'):
                            dirs.append(entry.name)
                    elif entry.is_file(follow_symlinks=False):
                        st = entry.stat(follow_symlinks=False)
                        files.append({'name': entry.name, 'size': st.st_size, 'mtime': st.st_mtime})
                except OSError:
                    continue   # vanished mid-listing / permission — just leave it out
    except OSError:
        pass
    if conn is not None and rel_folder.split('/', 1)[0] != '.media':
        _sync_direct_children(conn, rel_folder, files)
    return dirs, files


def _direct_rows(conn, rel_folder):
    """{path: (size, mtime)} of disk_files rows sitting directly in rel_folder."""
    if rel_folder:
        esc = glob_escape(rel_folder)
        cur = conn.execute('SELECT path, size, mtime FROM disk_files '
                           'WHERE path GLOB ? AND path NOT GLOB ?', (f'{esc}/*', f'{esc}/*/*'))
    else:
        cur = conn.execute("SELECT path, size, mtime FROM disk_files WHERE path NOT GLOB '*/*'")
    return {r[0]: (r[1], r[2]) for r in cur}


def _sync_direct_children(conn, rel_folder, files):
    present = {}
    for f in files:
        if media_kind(f['name']) is not None:
            present[_join(rel_folder, f['name'])] = (f['size'], f['mtime'])
    stored = _direct_rows(conn, rel_folder)
    changed = [(p, s, m) for p, (s, m) in present.items() if stored.get(p) != (s, m)]
    gone = [(p,) for p in stored if p not in present]
    if not changed and not gone:
        return
    now = int(time.time())
    conn.executemany('INSERT INTO disk_files (path, size, mtime, seen_at) VALUES (?, ?, ?, ?) '
                     'ON CONFLICT(path) DO UPDATE SET size = excluded.size, '
                     'mtime = excluded.mtime, seen_at = excluded.seen_at',
                     [(p, s, m, now) for p, s, m in changed])
    conn.executemany('DELETE FROM disk_files WHERE path = ?', gone)
    conn.commit()


def record_files(conn, entries):
    """Upsert (rel_path, size, mtime) rows for files the app itself just wrote (e.g.
    uploads), so rollups count them before the next full walk. No commit."""
    now = int(time.time())
    conn.executemany('INSERT INTO disk_files (path, size, mtime, seen_at) VALUES (?, ?, ?, ?) '
                     'ON CONFLICT(path) DO UPDATE SET size = excluded.size, '
                     'mtime = excluded.mtime, seen_at = excluded.seen_at',
                     [(p, s, m, now) for p, s, m in entries])


def subfolder_media_paths(conn, rel_folder):
    """Recursive rollup source for the immediate child folders of rel_folder:
    {child_name: [rel_path, ...]} of every disk_files row anywhere under each child.
    One GLOB range query (a full scan at the root, inherent to enumerating every
    top-level folder), bucketed by first path component in Python."""
    rel_folder = rel_folder.strip('/')
    if rel_folder:
        cur = conn.execute('SELECT path FROM disk_files WHERE path GLOB ?',
                           (f'{glob_escape(rel_folder)}/*',))
        base = len(rel_folder) + 1
    else:
        cur = conn.execute('SELECT path FROM disk_files')
        base = 0
    out = {}
    for (path,) in cur:
        rest = path[base:]
        slash = rest.find('/')
        if slash > 0:
            out.setdefault(rest[:slash], []).append(path)
    return out


def walk_media_files(data_root):
    """Yield (rel_path, size, mtime) for every regular MEDIA_EXTENSIONS file under
    data_root (no symlinks followed, top-level .media skipped entirely). Unreadable
    directories are skipped silently, as find would warn and carry on."""
    stack = ['']
    while stack:
        rel_dir = stack.pop()
        abs_dir = os.path.join(data_root, rel_dir) if rel_dir else data_root
        try:
            it = os.scandir(abs_dir)
        except OSError:
            continue
        with it:
            for entry in it:
                rel = _join(rel_dir, entry.name)
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if rel != '.media':
                            stack.append(rel)
                    elif entry.is_file(follow_symlinks=False) and media_kind(entry.name):
                        st = entry.stat(follow_symlinks=False)
                        yield rel, st.st_size, st.st_mtime
                except OSError:
                    continue


def run_inventory(conn, data_root, progress=None):
    """Full walk → replace disk_files. Rows are upserted with this run's seen_at in
    batches (readers keep seeing the previous inventory for not-yet-reached paths),
    then everything older is deleted and refreshed_at stamped in one final commit.
    progress(done) is called after each batch. Returns the number of files found."""
    stamp = int(time.time())
    _meta_set(conn, 'walk_started_at', stamp)
    conn.commit()
    sql = ('INSERT INTO disk_files (path, size, mtime, seen_at) VALUES (?, ?, ?, ?) '
           'ON CONFLICT(path) DO UPDATE SET size = excluded.size, mtime = excluded.mtime, '
           'seen_at = excluded.seen_at')
    batch, done = [], 0
    for rel, size, mtime in walk_media_files(data_root):
        batch.append((rel, size, mtime, stamp))
        if len(batch) >= _WALK_BATCH:
            conn.executemany(sql, batch)
            conn.commit()
            done += len(batch)
            batch = []
            if progress:
                progress(done)
    conn.executemany(sql, batch)
    done += len(batch)
    conn.execute('DELETE FROM disk_files WHERE seen_at < ?', (stamp,))
    _meta_set(conn, 'refreshed_at', int(time.time()))
    conn.commit()
    if progress:
        progress(done)
    return done


class InventoryJob:
    """The background full walk as a web job: {running, done, total, started_at,
    finished_at, error} like the other web jobs, one at a time per process.
    `db` is a database.ThreadLocalDB (the walk thread uses its own connection)."""

    def __init__(self, db, data_root):
        self.db = db
        self.data_root = data_root
        self.state = {'running': False, 'done': 0, 'total': 0, 'started_at': None,
                      'finished_at': None, 'error': None}
        self._lock = threading.Lock()

    def start(self, spawn):
        """Kick off the walk via spawn(target) (web's _spawn_job, or a plain call to run
        it synchronously). Returns False if one is already running."""
        with self._lock:
            if self.state['running']:
                return False
            prev = self.db.conn.execute('SELECT COUNT(*) FROM disk_files').fetchone()[0]
            self.state.update(running=True, done=0, total=prev, started_at=int(time.time()),
                              finished_at=None, error=None)

        def _progress(done):
            self.state['done'] = done
            self.state['total'] = max(self.state['total'], done)

        def _run():
            try:
                run_inventory(self.db.conn, self.data_root, _progress)
            except Exception as exc:  # hook-allow: broad — reported via the job's error field
                self.state['error'] = str(exc)
            finally:
                self.state['total'] = self.state['done']
                self.state['finished_at'] = int(time.time())
                self.state['running'] = False
        spawn(_run)
        return True

    def should_autostart(self):
        """Never completed, and no walk started (by any process) in the last STALE_AFTER."""
        conn = self.db.conn
        if refreshed_at(conn) is not None:
            return False
        started = _meta_get(conn, 'walk_started_at')
        return started is None or int(started) < time.time() - STALE_AFTER

    def status(self):
        conn = self.db.conn
        out = {k: self.state[k] for k in ('running', 'done', 'total', 'error',
                                          'started_at', 'finished_at')}
        out['refreshed_at'] = refreshed_at(conn)
        return out

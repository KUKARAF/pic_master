"""Run missing ML here" — the file manager's per-folder ML job.

The /bulk page runs one pipeline stage over the whole library; this runs every stage
a folder's files are missing, each only over the files missing it (per scan_status,
so the Files tab colours and this job agree on what "missing" means). The per-file
work itself is NOT here: web.py refactored each bulk job's loop into a function over
an explicit file list, and registers one Runner per stage that calls it. This module
is the model-free part — planning, the stage loop, cancel, status — so it can be
tested without loading anything.

Stage outcome policy (run_plan):
  * the runner's prepare() raises (model/backend can't be built, e.g. the GPU guard
    of the object index), or an `unavailable` exception (StageUnavailable, a remote
    WorkerError) escapes the stage → that stage is 'skipped' with the reason and the
    run continues with the next stage;
  * the equivalent /bulk job is running (claim() returns a message) → 'skipped';
  * any other exception escapes the stage — the bulk job would have stopped with this
    error too, e.g. object_index's MAX_CONSECUTIVE_EMBED_FAILURES abort → the stage is
    'error' and the whole run stops with that error (later stages stay 'pending').
Cancel is cooperative: the stage functions check should_stop() between files, so a
cancelled run stops after the file in hand; every write is per file, so nothing is
left half-written. The cancelled stage and every later one are marked 'cancelled'.
"""
import threading
import time

from . import scan_status

# Execution order: images first (embedding/detection before the indexes that reuse
# them — bodies reuse tags' person boxes), videos last with the face scan after the
# stills. Every scan_status stage appears exactly once.
STAGE_ORDER = ('exif', 'clip', 'tags', 'faces', 'bodies', 'objects', 'region', 'pattern',
               'phash', 'tiles', 'stills', 'video_faces')

# key -> (label, required, kind), from scan_status.STAGES.
STAGE_INFO = {key: (label, req, kind)
              for kind, stages in scan_status.STAGES.items()
              for key, label, req in stages}
assert set(STAGE_INFO) == set(STAGE_ORDER), 'folder_ml.STAGE_ORDER out of sync with scan_status'


class StageUnavailable(Exception):
    """Raised by a runner when its model/backend can't be used — skip the stage."""


class Runner:
    """One stage's hooks. prepare() → ctx (raise to skip the stage: backend
    unavailable); process(files, ctx, on_progress(done, failed), should_stop) →
    (done, failed), where files is [(file_id, rel_path)] in folder order."""

    def __init__(self, process, prepare=None):
        self.process = process
        self.prepare = prepare or (lambda: None)


class CountingLog:
    """Wraps an ErrorLog-like object (or a log(path, msg) function) and counts the
    calls — how the folder job counts failures of builders that only report them
    through the error log (body/tile/object index)."""

    def __init__(self, inner):
        self._log = inner.log if hasattr(inner, 'log') else inner
        self.count = 0

    def log(self, path, msg):
        self.count += 1
        self._log(path, msg)


def plan(files, statuses, stages=None):
    """[(key, [(file_id, rel_path)])] in STAGE_ORDER for the stages to run.

    `files` is [(file_id, rel_path)] (the folder's tracked files), `statuses` is
    scan_status.status_for_files over them. Broken files are never planned. `stages`
    None = every REQUIRED stage with at least one missing file (the optional tiles
    stage only when asked for by name); an explicit list runs exactly those stages
    (still only over files missing them; one with nothing missing is planned empty
    and finishes at once). Raises ValueError on an unknown key."""
    if stages is not None:
        unknown = [s for s in stages if s not in STAGE_INFO]
        if unknown:
            raise ValueError(f'unknown stage(s): {", ".join(map(str, unknown))}')
    missing = missing_by_stage(files, statuses)
    out = []
    for key in STAGE_ORDER:
        if stages is None:
            if STAGE_INFO[key][1] and missing[key]:
                out.append((key, missing[key]))
        elif key in stages:
            out.append((key, missing[key]))
    return out


def missing_by_stage(files, statuses):
    """{stage key: [(file_id, rel_path)]} of non-broken files missing each stage."""
    missing = {key: [] for key in STAGE_ORDER}
    for fid, rel in files:
        st = statuses.get(fid)
        if st is None or st['state'] == 'broken':
            continue
        for key, ok in st['stages'].items():
            if not ok:
                missing[key].append((fid, rel))
    return missing


class FolderMLJob:
    """State + cancel flag of the (single) folder ML job. status() is the
    GET /api/fm/ml/status body."""

    def __init__(self):
        self.lock = threading.Lock()
        self._cancel = threading.Event()
        self.state = {'running': False, 'path': None, 'started_at': None,
                      'finished_at': None, 'current_stage': None, 'error': None,
                      'cancelled': False, 'stages': []}

    def status(self):
        with self.lock:
            st = dict(self.state)
            st['stages'] = [dict(s) for s in st['stages']]
        return st

    def stage_running(self, key):
        """True while this job is processing stage `key` (read by the /bulk starts)."""
        return self.state['running'] and self.state['current_stage'] == key

    def try_begin(self, path, stage_plan):
        """Atomically claim the job for a run; False if one is already running."""
        with self.lock:
            if self.state['running']:
                return False
            self._cancel.clear()
            self.state.update(
                running=True, path=path, started_at=int(time.time()), finished_at=None,
                current_stage=None, error=None, cancelled=False,
                stages=[{'key': key, 'label': STAGE_INFO[key][0], 'total': len(files),
                         'done': 0, 'failed': 0, 'status': 'pending', 'message': None}
                        for key, files in stage_plan])
        return True

    def cancel(self):
        """Request a cooperative stop; False if nothing is running."""
        if not self.state['running']:
            return False
        self._cancel.set()
        return True

    def should_stop(self):
        return self._cancel.is_set()

    def finish(self, error=None):
        with self.lock:
            self.state.update(running=False, current_stage=None,
                              finished_at=int(time.time()))
            if error is not None and self.state['error'] is None:
                self.state['error'] = error


def run_plan(job, stage_plan, runners, claim=None, invalidate=None,
             unavailable=(StageUnavailable,)):
    """Run `stage_plan` (from plan()) stage by stage, updating job.state['stages'].

    claim(key) → None to go ahead, or a message to skip the stage (its /bulk job is
    running); it is called with job.state['current_stage'] already set to `key`, so a
    /bulk start that sets its own running flag and THEN checks stage_running() can't
    slip past it (one of the two always sees the other). invalidate() (scan_status's)
    runs after each stage that ran. Does not call job.finish(); returns the run's
    error message or None."""
    entries = {s['key']: s for s in job.state['stages']}
    error = None
    for key, files in stage_plan:
        if job.should_stop():
            break
        entry = entries[key]
        with job.lock:
            job.state['current_stage'] = key
        try:
            error = _run_stage(job, entry, files, runners.get(key), claim, invalidate,
                               unavailable)
        finally:
            with job.lock:
                job.state['current_stage'] = None
        if error is not None or entry['status'] == 'cancelled':
            break
    if job.should_stop():
        job.state['cancelled'] = True
        for entry in job.state['stages']:
            if entry['status'] in ('pending', 'running'):
                entry['status'] = 'cancelled'
    return error


def _run_stage(job, entry, files, runner, claim, invalidate, unavailable):
    """One stage of run_plan; sets entry's status. Returns the run error or None."""
    if runner is None:
        entry.update(status='skipped', message='Not supported by this server.')
        return None
    busy = claim(entry['key']) if claim is not None else None
    if busy:
        entry.update(status='skipped', message=busy)
        return None
    if not files:
        entry['status'] = 'done'
        return None
    entry['status'] = 'running'
    try:
        ctx = runner.prepare()
    except Exception as exc:  # hook-allow: broad — any backend failure means "unavailable"; reported
        entry.update(status='skipped', message=f'Unavailable: {exc}')
        return None

    def _progress(done, failed):
        entry['done'] = done
        entry['failed'] = failed

    try:
        done, failed = runner.process(files, ctx, _progress, job.should_stop)
        entry.update(done=done, failed=failed)
    except unavailable as exc:
        entry.update(status='skipped', message=f'Unavailable: {exc}')
        return None
    except Exception as exc:  # hook-allow: broad — stage abort; surfaced in status, run stops
        msg = str(exc) or type(exc).__name__
        entry.update(status='error', message=msg)
        return f'{entry["label"]}: {msg}'
    finally:
        if invalidate is not None:
            invalidate()
    entry['status'] = 'cancelled' if job.should_stop() and done < len(files) else 'done'
    return None

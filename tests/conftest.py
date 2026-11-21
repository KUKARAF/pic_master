"""Test-suite plumbing.

Many tests build their fixtures with ``tempfile.mkdtemp()`` and never remove the result,
so every full run used to leave ~100 ``/tmp/tmpXXXXXXXX`` directories (each with a
media.db / manual.db, ~2.7 MB) behind. /tmp is a RAM-backed tmpfs on the dev box: a dozen
runs filled it and later tests failed with ``Disk quota exceeded`` / ``disk I/O error``.

Fix at the source for the whole suite: point ``tempfile`` at one per-session directory and
delete it when the session ends, so nothing outlives the run no matter how a test makes its
temp paths.
"""
import shutil
import tempfile

import pytest


@pytest.fixture(scope='session', autouse=True)
def _contain_tempfiles():
    root = tempfile.mkdtemp(prefix='mm-tests-')
    previous = tempfile.tempdir
    tempfile.tempdir = root
    try:
        yield root
    finally:
        tempfile.tempdir = previous
        shutil.rmtree(root, ignore_errors=True)

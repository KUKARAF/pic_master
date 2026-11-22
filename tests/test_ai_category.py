"""Photos in the `ai` category never act as a face-match reference (unless MEDIA_INCLUDE_AI=1).

Run: python -m pytest tests/test_ai_category.py
"""
import sys
import tempfile
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault('cv2', MagicMock())

from test_person_cleanup import World, NAME  # noqa: E402


@pytest.fixture
def w():
    with tempfile.TemporaryDirectory() as tmp:
        yield World(tmp)


def test_ai_category_photo_is_not_a_match_reference(w, monkeypatch):
    m = w.manual
    fid, _src = w.promoted(w.alex, auto_matched=False)
    checksum = w.manual.get_face(fid)['checksum']
    assert m.find_matching_identity(w.alex.tobytes())[0] == NAME
    ai = m.create_category('AI')                      # name match is case-insensitive
    m.add_file_category(checksum, ai)
    assert m.get_ai_checksums() == {checksum}
    assert m.find_matching_identity(w.alex.tobytes())[0] is None      # cache invalidated on assignment
    m.remove_file_category(checksum, ai)
    assert m.find_matching_identity(w.alex.tobytes())[0] == NAME      # and again on removal


def test_other_categories_do_not_matter(w):
    m = w.manual
    fid, _src = w.promoted(w.alex, auto_matched=False)
    checksum = w.manual.get_face(fid)['checksum']
    m.add_file_category(checksum, m.create_category('portraits'))
    assert m.find_matching_identity(w.alex.tobytes())[0] == NAME


def test_include_ai_env_flag(w, monkeypatch):
    monkeypatch.setenv('MEDIA_INCLUDE_AI', '1')
    m = w.manual
    fid, _src = w.promoted(w.alex, auto_matched=False)
    checksum = w.manual.get_face(fid)['checksum']
    m.add_file_category(checksum, m.create_category('ai'))
    assert m.find_matching_identity(w.alex.tobytes())[0] == NAME

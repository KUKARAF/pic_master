"""A GPU that runs out of memory / gets lost stops a bulk job cleanly.

After CL_OUT_OF_RESOURCES / DEVICE_LOST the driver is unreliable until restart, so a
bulk job must not keep going and log one failure per remaining file: it stops on the
first GPU-fatal per-file error with "The GPU ran out of memory or was lost … Restart
the server", leaving the rest unprocessed for the next run. Ordinary per-file errors
(a bad image) still just get logged while the job continues.

Run: python tests/test_gpu_fatal.py   (or python -m pytest tests/test_gpu_fatal.py)
"""
import os
import tempfile
import time

import xxhash
from fastapi.testclient import TestClient
from PIL import Image

import media_manager.web as web_mod
from media_manager import compute
from media_manager.web import create_app

OCL = ('[ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running '
       'OpenVINO-EP-subgraph_1 node.\nException from ocl_common.hpp:56:\n[GPU] CL_OUT_OF_RESOURCES exception.')


class FakeDetector:
    _model_name = 'fake'

    def __init__(self, fail_from):
        self.fail_from, self.seen = fail_from, []

    @staticmethod
    def model_id(_name):
        return 'fake'

    def detect_faces(self, paths):
        out = []
        for p in paths:
            self.seen.append(os.path.basename(p))
            name = os.path.basename(p)
            if name == 'bad.jpg':
                out.append((p, [], 'cv2.imread returned None'))      # ordinary failure
            elif len(self.seen) > self.fail_from:
                out.append((p, [], OCL))                               # GPU gone
            else:
                out.append((p, [], None))
        return out


def _wait(client):
    for _ in range(400):
        s = client.get('/api/detect-faces/status').json()
        if not s['running']:
            return s
        time.sleep(0.05)
    raise AssertionError('job did not finish')


def run():
    assert compute.is_gpu_fatal(OCL) and compute.is_gpu_fatal('Native API returns: 20 (UR_RESULT_ERROR_DEVICE_LOST)')
    assert not compute.is_gpu_fatal('cv2.imread returned None')
    assert compute.job_error_text(ValueError('nope')) == 'nope'
    assert compute.job_error_text(RuntimeError(OCL)).startswith('The GPU ran out of memory')

    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, '.media'))
    names = ['a.jpg', 'bad.jpg', 'c.jpg', 'd.jpg', 'e.jpg', 'f.jpg']
    for i, n in enumerate(names):
        Image.new('RGB', (16, 16), (i * 30, 0, 0)).save(os.path.join(root, n))
    app = create_app(root)
    db = app.state.dbs[0]
    for n in names:
        data = open(os.path.join(root, n), 'rb').read()
        db.upsert_file_path(n, xxhash.xxh64(data).hexdigest(), size=len(data), modified_time=1)
    db.conn.commit()
    client = TestClient(app)

    # Ordinary per-file errors: logged, job completes.
    det = FakeDetector(fail_from=99)
    web_mod._get_face_detector = lambda: det
    client.post('/api/detect-faces/start')
    s = _wait(client)
    assert s['error'] is None and s['done'] == 6, s

    # GPU-fatal on the 3rd file: job stops right there with the restart hint.
    db.conn.execute('DELETE FROM faces')
    db.conn.commit()
    det = FakeDetector(fail_from=2)
    web_mod._get_face_detector = lambda: det
    client.post('/api/detect-faces/start')
    s = _wait(client)
    assert s['error'] and s['error'].startswith('The GPU ran out of memory or was lost'), s
    assert 'CL_OUT_OF_RESOURCES' in s['error'] or 'ONNXRuntimeError' in s['error'], s
    assert s['done'] == 2, s              # stopped at the 3rd file (the GPU error), not counted done
    assert s['pending'] >= 3, s                                # the rest is left for next run
    print('OK  ordinary errors keep going; GPU-fatal stops the job with the restart hint')
    print('ALL ASSERTIONS PASSED')


def test_gpu_fatal():
    run()


if __name__ == '__main__':
    run()

"""Rotation-invariant face detection with PCN (Progressive Calibration Networks,
CVPR 2018), running on onnxruntime — NO PyTorch.

This is a faithful, trimmed vendoring of siriusdemon/pytorch-PCN (BSD-2-Clause,
see LICENSE beside this file). The pyramid / 3-stage cascade / calibration / NMS
logic is copied verbatim; the only changes are the two places that touched torch:

  * set_input() returns a contiguous float32 NCHW numpy array instead of a
    torch.FloatTensor;
  * each `net(x)` forward becomes `session.run(None, {'x': x})` on an
    onnxruntime.InferenceSession (outputs are (cls, rotate, bbox), matching the
    original models' forward order).

The three tiny nets (pcn1/2/3.onnx, ~2 MB total) were exported once from the
upstream BSD-2 .pth weights and produce detections numerically identical to the
reference. Video-only smoothing and the drawing helpers are dropped.

PCN detects faces at any in-plane rotation and returns, per face, an axis-aligned
square box plus the in-plane angle (degrees). `crop_face` warps that rotated square
upright — exactly the crop to hand to InsightFace for a correctly-aligned embedding.
"""
import os
import numpy as np
import cv2

# --- global settings (unchanged from upstream) ---------------------------------
EPS = 1e-5
minFace_ = 20 * 1.4
scale_ = 1.414
stride_ = 8
classThreshold_ = [0.37, 0.43, 0.97]
nmsThreshold_ = [0.8, 0.8, 0.3]
angleRange_ = 45


class Window2:
    def __init__(self, x, y, w, h, angle, scale, conf):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.angle = angle
        self.scale = scale
        self.conf = conf


class Window:
    """Final detection: axis-aligned square (x, y, width) + in-plane `angle`
    (degrees) + `score`."""
    def __init__(self, x, y, width, angle, score):
        self.x = x
        self.y = y
        self.width = width
        self.angle = angle
        self.score = score


def preprocess_img(img, dim=None):
    if dim:
        img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_NEAREST)
    return img - np.array([104, 117, 123])


def resize_img(img, scale):
    h, w = img.shape[:2]
    h_, w_ = int(h / scale), int(w / scale)
    img = img.astype(np.float32)
    return cv2.resize(img, (w_, h_), interpolation=cv2.INTER_NEAREST)


def pad_img(img):
    row = min(int(img.shape[0] * 0.2), 100)
    col = min(int(img.shape[1] * 0.2), 100)
    return cv2.copyMakeBorder(img, row, row, col, col, cv2.BORDER_CONSTANT)


def legal(x, y, img):
    return 0 <= x < img.shape[1] and 0 <= y < img.shape[0]


def inside(x, y, rect):
    return rect.x <= x < (rect.x + rect.w) and rect.y <= y < (rect.y + rect.h)


def IoU(w1, w2):
    xOverlap = max(0, min(w1.x + w1.w - 1, w2.x + w2.w - 1) - max(w1.x, w2.x) + 1)
    yOverlap = max(0, min(w1.y + w1.h - 1, w2.y + w2.h - 1) - max(w1.y, w2.y) + 1)
    intersection = xOverlap * yOverlap
    unio = w1.w * w1.h + w2.w * w2.h - intersection
    return intersection / unio


def NMS(winlist, local, threshold):
    length = len(winlist)
    if length == 0:
        return winlist
    winlist.sort(key=lambda x: x.conf, reverse=True)
    flag = [0] * length
    for i in range(length):
        if flag[i]:
            continue
        for j in range(i + 1, length):
            if local and abs(winlist[i].scale - winlist[j].scale) > EPS:
                continue
            if IoU(winlist[i], winlist[j]) > threshold:
                flag[j] = 1
    return [winlist[i] for i in range(length) if not flag[i]]


def deleteFP(winlist):
    length = len(winlist)
    if length == 0:
        return winlist
    winlist.sort(key=lambda x: x.conf, reverse=True)
    flag = [0] * length
    for i in range(length):
        if flag[i]:
            continue
        for j in range(i + 1, length):
            win = winlist[j]
            if inside(win.x, win.y, winlist[i]) and inside(win.x + win.w - 1, win.y + win.h - 1, winlist[i]):
                flag[j] = 1
    return [winlist[i] for i in range(length) if not flag[i]]


def set_input(img):
    """torch-free: contiguous float32 NCHW numpy array for onnxruntime."""
    if isinstance(img, list):
        img = np.stack(img, axis=0)
    else:
        img = img[np.newaxis, :, :, :]
    img = img.transpose((0, 3, 1, 2))
    return np.ascontiguousarray(img, dtype=np.float32)


def _run(net, x):
    """Forward pass through an onnxruntime session; returns (cls, rotate, bbox)."""
    return net.run(None, {'x': x})


def trans_window(img, imgPad, winlist):
    row = (imgPad.shape[0] - img.shape[0]) // 2
    col = (imgPad.shape[1] - img.shape[1]) // 2
    ret = []
    for win in winlist:
        if win.w > 0 and win.h > 0:
            ret.append(Window(win.x - col, win.y - row, win.w, win.angle, win.conf))
    return ret


def stage1(img, imgPad, net, thres):
    row = (imgPad.shape[0] - img.shape[0]) // 2
    col = (imgPad.shape[1] - img.shape[1]) // 2
    winlist = []
    netSize = 24
    curScale = minFace_ / netSize
    img_resized = resize_img(img, curScale)
    while min(img_resized.shape[:2]) >= netSize:
        img_resized = preprocess_img(img_resized)
        cls_prob, rotate, bbox = _run(net, set_input(img_resized))
        w = netSize * curScale
        for i in range(cls_prob.shape[2]):
            for j in range(cls_prob.shape[3]):
                if cls_prob[0, 1, i, j] > thres:
                    sn = bbox[0, 0, i, j]
                    xn = bbox[0, 1, i, j]
                    yn = bbox[0, 2, i, j]
                    rx = int(j * curScale * stride_ - 0.5 * sn * w + sn * xn * w + 0.5 * w) + col
                    ry = int(i * curScale * stride_ - 0.5 * sn * w + sn * yn * w + 0.5 * w) + row
                    rw = int(w * sn)
                    if legal(rx, ry, imgPad) and legal(rx + rw - 1, ry + rw - 1, imgPad):
                        if rotate[0, 1, i, j] > 0.5:
                            winlist.append(Window2(rx, ry, rw, rw, 0, curScale, float(cls_prob[0, 1, i, j])))
                        else:
                            winlist.append(Window2(rx, ry, rw, rw, 180, curScale, float(cls_prob[0, 1, i, j])))
        img_resized = resize_img(img_resized, scale_)
        curScale = img.shape[0] / img_resized.shape[0]
    return winlist


def stage2(img, img180, net, thres, dim, winlist):
    length = len(winlist)
    if length == 0:
        return winlist
    datalist = []
    height = img.shape[0]
    for win in winlist:
        if abs(win.angle) < EPS:
            datalist.append(preprocess_img(img[win.y:win.y + win.h, win.x:win.x + win.w, :], dim))
        else:
            y2 = win.y + win.h - 1
            y = height - 1 - y2
            datalist.append(preprocess_img(img180[y:y + win.h, win.x:win.x + win.w, :], dim))
    cls_prob, rotate, bbox = _run(net, set_input(datalist))
    ret = []
    for i in range(length):
        if cls_prob[i, 1] > thres:
            sn = bbox[i, 0]
            xn = bbox[i, 1]
            yn = bbox[i, 2]
            cropX = winlist[i].x
            cropY = winlist[i].y
            cropW = winlist[i].w
            if abs(winlist[i].angle) > EPS:
                cropY = height - 1 - (cropY + cropW - 1)
            w = int(sn * cropW)
            x = int(cropX - 0.5 * sn * cropW + cropW * sn * xn + 0.5 * cropW)
            y = int(cropY - 0.5 * sn * cropW + cropW * sn * yn + 0.5 * cropW)
            maxRotateScore = 0
            maxRotateIndex = 0
            for j in range(3):
                if rotate[i, j] > maxRotateScore:
                    maxRotateScore = rotate[i, j]
                    maxRotateIndex = j
            if legal(x, y, img) and legal(x + w - 1, y + w - 1, img):
                if abs(winlist[i].angle) < EPS:
                    if maxRotateIndex == 0:
                        angle = 90
                    elif maxRotateIndex == 1:
                        angle = 0
                    else:
                        angle = -90
                    ret.append(Window2(x, y, w, w, angle, winlist[i].scale, float(cls_prob[i, 1])))
                else:
                    if maxRotateIndex == 0:
                        angle = 90
                    elif maxRotateIndex == 1:
                        angle = 180
                    else:
                        angle = -90
                    ret.append(Window2(x, height - 1 - (y + w - 1), w, w, angle, winlist[i].scale, float(cls_prob[i, 1])))
    return ret


def stage3(imgPad, img180, img90, imgNeg90, net, thres, dim, winlist):
    length = len(winlist)
    if length == 0:
        return winlist
    datalist = []
    height, width = imgPad.shape[:2]
    for win in winlist:
        if abs(win.angle) < EPS:
            datalist.append(preprocess_img(imgPad[win.y:win.y + win.h, win.x:win.x + win.w, :], dim))
        elif abs(win.angle - 90) < EPS:
            datalist.append(preprocess_img(img90[win.x:win.x + win.w, win.y:win.y + win.h, :], dim))
        elif abs(win.angle + 90) < EPS:
            x = win.y
            y = width - 1 - (win.x + win.w - 1)
            datalist.append(preprocess_img(imgNeg90[y:y + win.h, x:x + win.w, :], dim))
        else:
            y2 = win.y + win.h - 1
            y = height - 1 - y2
            datalist.append(preprocess_img(img180[y:y + win.h, win.x:win.x + win.w], dim))
    cls_prob, rotate, bbox = _run(net, set_input(datalist))
    ret = []
    for i in range(length):
        if cls_prob[i, 1] > thres:
            sn = bbox[i, 0]
            xn = bbox[i, 1]
            yn = bbox[i, 2]
            cropX = winlist[i].x
            cropY = winlist[i].y
            cropW = winlist[i].w
            img_tmp = imgPad
            if abs(winlist[i].angle - 180) < EPS:
                cropY = height - 1 - (cropY + cropW - 1)
                img_tmp = img180
            elif abs(winlist[i].angle - 90) < EPS:
                cropX, cropY = cropY, cropX
                img_tmp = img90
            elif abs(winlist[i].angle + 90) < EPS:
                cropX = winlist[i].y
                cropY = width - 1 - (winlist[i].x + winlist[i].w - 1)
                img_tmp = imgNeg90
            w = int(sn * cropW)
            x = int(cropX - 0.5 * sn * cropW + cropW * sn * xn + 0.5 * cropW)
            y = int(cropY - 0.5 * sn * cropW + cropW * sn * yn + 0.5 * cropW)
            angle = angleRange_ * float(rotate[i, 0])
            if legal(x, y, img_tmp) and legal(x + w - 1, y + w - 1, img_tmp):
                if abs(winlist[i].angle) < EPS:
                    ret.append(Window2(x, y, w, w, angle, winlist[i].scale, float(cls_prob[i, 1])))
                elif abs(winlist[i].angle - 180) < EPS:
                    ret.append(Window2(x, height - 1 - (y + w - 1), w, w, 180 - angle, winlist[i].scale, float(cls_prob[i, 1])))
                elif abs(winlist[i].angle - 90) < EPS:
                    ret.append(Window2(y, x, w, w, 90 - angle, winlist[i].scale, float(cls_prob[i, 1])))
                else:
                    ret.append(Window2(width - y - w, x, w, w, -90 + angle, winlist[i].scale, float(cls_prob[i, 1])))
    return ret


def _detect(img, imgPad, nets):
    img180 = cv2.flip(imgPad, 0)
    img90 = cv2.transpose(imgPad)
    imgNeg90 = cv2.flip(img90, 0)
    winlist = stage1(img, imgPad, nets[0], classThreshold_[0])
    winlist = NMS(winlist, True, nmsThreshold_[0])
    winlist = stage2(imgPad, img180, nets[1], classThreshold_[1], 24, winlist)
    winlist = NMS(winlist, True, nmsThreshold_[1])
    winlist = stage3(imgPad, img180, img90, imgNeg90, nets[2], classThreshold_[2], 48, winlist)
    winlist = NMS(winlist, False, nmsThreshold_[2])
    winlist = deleteFP(winlist)
    return winlist


def pcn_detect(img, nets):
    imgPad = pad_img(img)
    winlist = _detect(img, imgPad, nets)
    return trans_window(img, imgPad, winlist)


def rotate_point(x, y, centerX, centerY, angle):
    x -= centerX
    y -= centerY
    theta = -angle * np.pi / 180
    rx = int(centerX + x * np.cos(theta) - y * np.sin(theta))
    ry = int(centerY + x * np.sin(theta) + y * np.cos(theta))
    return rx, ry


def crop_face(img, face, crop_size=200):
    """Warp the rotated square `face` (a Window) to an upright `crop_size` square.
    Returns (crop, pointlist) where pointlist is the 4 rotated corners in the
    original image's coordinates."""
    x1, y1 = face.x, face.y
    x2, y2 = face.width + face.x - 1, face.width + face.y - 1
    centerX, centerY = (x1 + x2) // 2, (y1 + y2) // 2
    lst = (x1, y1), (x1, y2), (x2, y2), (x2, y1)
    pointlist = [rotate_point(x, y, centerX, centerY, face.angle) for x, y in lst]
    srcTriangle = np.array([pointlist[0], pointlist[1], pointlist[2]], dtype=np.float32)
    dstTriangle = np.array([(0, 0), (0, crop_size - 1), (crop_size - 1, crop_size - 1)], dtype=np.float32)
    rotMat = cv2.getAffineTransform(srcTriangle, dstTriangle)
    ret = cv2.warpAffine(img, rotMat, (crop_size, crop_size))
    return ret, pointlist


class PCNDetector:
    """Lazy-loading rotation-invariant face detector. detect() returns
    [{'bbox':[x1,y1,x2,y2], 'angle':degrees, 'conf':float, 'points':[(x,y)x4]}]."""

    def __init__(self, model_dir=None):
        import onnxruntime as ort
        from .. import compute
        d = model_dir or os.path.dirname(os.path.abspath(__file__))
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)
        # Route through the central provider pick so PCN rides the same accelerator
        # as InsightFace (OpenVINO GPU on Intel Arc, CUDA on NVIDIA) rather than the
        # old hardcoded CPU-only list. Tiny cascade nets, so the win is small — this
        # is mostly for consistency; the intra_op thread tuning above is harmless
        # (simply ignored) when a GPU provider binds.
        providers = compute.onnx_providers()
        self.nets = [
            ort.InferenceSession(os.path.join(d, f'pcn{i}.onnx'),
                                 sess_options=so, providers=providers)
            for i in (1, 2, 3)
        ]

    def detect(self, bgr_image):
        wins = pcn_detect(bgr_image, self.nets)
        out = []
        for w in wins:
            _crop, pts = crop_face(bgr_image, w, 2)  # cheap: just want the corners
            out.append({
                'bbox': [float(w.x), float(w.y), float(w.x + w.width), float(w.y + w.width)],
                'angle': float(w.angle),
                'conf': float(w.score),
                'points': pts,
            })
        return out

    def upright_crop(self, bgr_image, win, crop_size=200):
        """Warp a detected face upright for embedding. `win` may be a Window or a
        dict from detect()."""
        if isinstance(win, dict):
            x1, y1, x2, _y2 = win['bbox']
            win = Window(int(x1), int(y1), int(x2 - x1), win['angle'], win['conf'])
        crop, _pts = crop_face(bgr_image, win, crop_size)
        return crop

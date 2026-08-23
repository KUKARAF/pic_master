"""Perceptual hashing for near-duplicate detection (Phase 1).

Two 64-bit hashes per image, computed with numpy only (no new dependency):

- pHash (DCT): downscale to 32x32 grayscale, take the 2-D DCT, keep the top-left
  8x8 low-frequency block, and threshold each coefficient against the block median.
  Low-frequency DCT coefficients are exactly what JPEG preserves under quality loss,
  so pHash is robust to re-encoding, resizing, and mild damage — the "same photo,
  different file" cases the duplicates detector cares about.
- dHash (difference): 9x8 grayscale, compare adjacent columns. A cheap, independent
  second opinion that catches the occasional pHash collision on flat/low-detail images.

Hashes are plain Python ints here; callers store them as 8-byte big-endian BLOBs
(int.to_bytes(8, 'big')) because a 64-bit value with the top bit set overflows
SQLite's signed INTEGER range.
"""

import numpy as np
from PIL import Image

# Bump when the algorithm changes so a rehash can be told apart from old rows.
ALGO_TAG = 'dct32-phash+dhash-v1'

_N = 32  # pHash works on a 32x32 grayscale downscale before the DCT.

# DCT-II basis matrix, precomputed once. Applying it as `_DCT @ img @ _DCT.T` is a
# 2-D DCT; we only care about the relative ordering of coefficients (for the median
# threshold), so the exact orthonormal scaling doesn't matter.
_k = np.arange(_N)
_DCT = np.cos(np.pi * (2 * _k[:, None] + 1) * _k[None, :] / (2 * _N)).astype(np.float32)


def _bits_to_int(bits) -> int:
    """Pack a flat boolean array (MSB first) into a Python int."""
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def phash(gray32: np.ndarray) -> int:
    """64-bit DCT perceptual hash from a 32x32 float32 grayscale array."""
    coeffs = _DCT @ gray32 @ _DCT.T
    block = coeffs[:8, :8].flatten()          # 64 low-frequency coefficients
    med = np.median(block[1:])                # exclude the DC term from the threshold
    return _bits_to_int(block > med)          # 64 bits


def dhash(gray9x8: np.ndarray) -> int:
    """64-bit difference hash from a 9x8 (rows x cols) float32 grayscale array."""
    diff = gray9x8[:, 1:] > gray9x8[:, :-1]   # 8x8 = 64 comparisons
    return _bits_to_int(diff.flatten())


def compute_hashes(pil_image: Image.Image):
    """(phash:int, dhash:int) for a PIL image (typically the cached 400px thumbnail —
    plenty of detail once reduced to 32x32/9x8)."""
    gray = pil_image.convert('L')
    g32 = np.asarray(gray.resize((_N, _N), Image.LANCZOS), dtype=np.float32)
    g98 = np.asarray(gray.resize((9, 8), Image.LANCZOS), dtype=np.float32)
    return phash(g32), dhash(g98)


def hamming(a: int, b: int) -> int:
    """Bit distance between two 64-bit hashes."""
    return bin(a ^ b).count('1')

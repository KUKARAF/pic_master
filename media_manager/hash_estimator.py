"""
Live progress decorator for hashing operations.
Prints: done, left, rate (files/min), ETA
"""
import time
import shutil
import sys

class HashEstimator:
    """
    Context manager / decorator that prints live progress while hashing files.
    Example:
        with HashEstimator(total=123) as est:
            for file_id, file_path in batch:
                do_hash(file_path)
                est.update(done=1)
    """
    def __init__(self, total=0, update_every=0.5):
        self.total = total
        self.done = 0
        self.start = time.time()
        self.update_every = update_every
        self.last_print = 0

    def __enter__(self):
        self._print()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # final line
        self._print(final=True)

    def update(self, done=1):
        self.done += done
        now = time.time()
        if now - self.last_print >= self.update_every:
            self._print()
            self.last_print = now

    def rate(self):
        elapsed = (time.time() - self.start) / 60.0
        return self.done / elapsed if elapsed else 0.0

    def eta(self):
        rate = self.rate()
        if rate == 0:
            return None
        left = self.total - self.done
        return left / rate  # minutes

    def _print(self, final=False):
        rate = self.rate()
        eta_min = self.eta()
        eta_str = f"{eta_min:.1f}min" if eta_min is not None else "--"
        # simple terminal line, overwrite with \r
        print(f"\r  {self.done}/{self.total}  {rate:.1f}/min  ETA {eta_str}", end="" if not final else "\n")
        sys.stdout.flush()

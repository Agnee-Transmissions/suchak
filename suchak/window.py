import numba as nb
import numpy as np

from suchak.jitclass import jitclass


@jitclass
class Window:
    offset: nb.int32
    period: nb.int32

    _buf: nb.double[:]
    _idx: nb.int32 = 0

    def __init__(self, period: int):
        self.offset = period - 1
        self.period = period

        self._buf = np.empty(period)
        self._buf[:] = np.nan

    def next(self, x: float) -> np.ndarray:
        self._buf[self._idx] = x
        self._idx = (self._idx + 1) % self.period
        return self._buf

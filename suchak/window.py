import numba as nb
import numpy as np

from suchak.util import jitclass


@jitclass
class Window:
    offset: nb.int32
    period: nb.int32

    _buf: nb.double[:]
    _idx: nb.int32

    def __init__(self, period: int):
        self.offset = period - 1
        self.period = period

        self._buf = np.empty(period)
        self._buf[:] = np.nan
        self._idx = 0

    def next(self, x: float) -> np.ndarray:
        self._buf[self._idx] = x
        self._idx = (self._idx + 1) % self.period
        return self._buf

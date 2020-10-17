import numba as nb
import numpy as np

from suchak.jitclass import jitclass


@jitclass
class Window:
    offset: nb.int32
    period: nb.int32
    buf: nb.float64[:]
    idx: nb.int32 = 0

    def __init__(self, period: int):
        self.offset = period - 1
        self.period = period

        self.buf = np.empty(period)
        self.buf[:] = np.nan

    def next(self, x: float) -> np.ndarray:
        self.buf[self.idx] = x
        self.idx = (self.idx + 1) % self.period
        return self.buf

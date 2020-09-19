import numba as nb
import numpy as np

from suchak.util import jitclass


@jitclass
class Tr:
    offset: nb.int32

    _c1: nb.double

    def __init__(self):
        self.offset = 0
        self._c1 = np.nan

    def next(self, h: float, l: float, c: float) -> float:
        if np.isnan(self._c1):
            ret = h - l
        else:
            ret = max(h - l, abs(h - self._c1), abs(l - self._c1))
        self._c1 = c
        return ret

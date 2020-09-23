import numba as nb
import numpy as np

from suchak.jitclass import jitclass


@jitclass
class TR:
    offset: nb.int32 = 0

    _c1: nb.double = np.nan

    def next(self, h: float, l: float, c: float) -> float:
        if np.isnan(self._c1):
            ret = h - l
        else:
            ret = max(h - l, abs(h - self._c1), abs(l - self._c1))
        self._c1 = c
        return ret

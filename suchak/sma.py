import numba as nb
import numpy as np

from suchak.jitclass import jitclass
from suchak.window import Window


@jitclass
class SMA:
    offset: nb.int32
    period: nb.int32

    _win: Window

    def __init__(self, period: int):
        self.period = period

        self._win = Window(period)
        self.offset = self._win.offset

    def next(self, x: float) -> float:
        return np.sum(self._win.next(x)) / self.period

    def next_arr(self, x_arr):
        out_len = len(x_arr)
        ret = np.empty(out_len)
        for i in range(out_len):
            ret[i] = self.next(x_arr[i])
        return ret

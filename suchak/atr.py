import numba as nb
import numpy as np

from suchak.ema import EMA
from suchak.jitclass import jitclass
from suchak.rma import RMA
from suchak.tr import TR


@jitclass
class ATR:
    offset: nb.int32
    period: nb.int32

    _rma: EMA
    _tr: TR

    def __init__(self, period: int):
        self.period = period

        self._rma = RMA(period)
        self._tr = TR()

        self.offset = self._rma.offset + self._tr.offset

    def next(self, h: float, l: float, c: float) -> float:
        return self._rma.next(self._tr.next(h, l, c))

    def next_arr(self, h_arr, l_arr, c_arr):
        out_len = len(c_arr)
        ret = np.empty(out_len)
        for i in range(out_len):
            ret[i] = self.next(h_arr[i], l_arr[i], c_arr[i])
        return ret

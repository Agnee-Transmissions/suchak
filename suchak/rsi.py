import numba as nb
import numpy as np

from suchak.ema import EMA
from suchak.jitclass import jitclass
from suchak.rma import RMA


@jitclass
class RSI:
    offset: nb.int32
    period: nb.int32

    _rma_up: EMA
    _rma_dn: EMA
    _c1: nb.double = np.nan

    def __init__(self, period: int):
        self.period = period

        self._rma_up = RMA(period)
        self._rma_dn = RMA(period)

        self.offset = self._rma_up.offset

    def next(self, c: float):
        up = max(c - self._c1, 0)
        dn = max(self._c1 - c, 0)

        dn_next = self._rma_dn.next(dn)
        if dn_next == 0:
            rs = np.nan
        else:
            rs = self._rma_up.next(up) / dn_next
        res = rs / (1 + rs) * 100

        self._c1 = c

        return res

    def next_arr(self, c_arr):
        out_len = len(c_arr)
        ret = np.empty(out_len)
        for i in range(out_len):
            ret[i] = self.next(c_arr[i])
        return ret

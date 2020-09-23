import numba as nb

from suchak.ema import EMA
from suchak.jitclass import jitclass
import numpy as np


@jitclass
class RSI:
    offset: nb.int32
    period: nb.int32

    _rma_up: EMA
    _rma_dn: EMA
    _c1: nb.double = np.nan

    def __init__(self, period: int = 14):
        self.period = period

        self._rma_up = EMA(period)
        self._rma_up._alpha = 1 / period

        self._rma_dn = EMA(period)
        self._rma_dn._alpha = 1 / period

        self.offset = self._rma_up.offset

    def next(self, c: float):
        up = max(c - self._c1, 0)
        dn = max(self._c1 - c, 0)
        rs = self._rma_up.next(up) / self._rma_dn.next(dn)
        res = rs / (1 + rs) * 100

        self._c1 = c

        return res

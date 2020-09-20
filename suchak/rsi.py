import numba as nb

from suchak.ema import EMA
from suchak.util import jitclass


@jitclass
class RSI:
    offset: nb.int32
    period: nb.int32

    _rma_up = EMA.class_type.instance_type
    _c1: nb.double

    def __init__(self, period: int):
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
        return rs / (1 + rs) * 100
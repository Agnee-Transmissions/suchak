import numba as nb

from suchak.sma import SMA
from suchak.tr import TR
from suchak.jitclass import jitclass


@jitclass
class ATR:
    offset: nb.int32
    period: nb.int32

    _sma: SMA
    _tr: TR

    def __init__(self, period: int):
        self.period = period

        self._sma = SMA(period)
        self._tr = TR()

        self.offset = self._sma.offset + self._tr.offset

    def next(self, h: float, l: float, c: float) -> float:
        return self._sma.next(self._tr.next(h, l, c))

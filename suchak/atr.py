import numba as nb

from suchak.sma import Sma
from suchak.tr import Tr
from suchak.util import jitclass


@jitclass
class Atr:
    offset: nb.int32
    period: nb.int32

    _sma: Sma.class_type.instance_type
    _tr: Tr.class_type.instance_type

    def __init__(self, period: int):
        self.period = period

        self._sma = Sma(period)
        self._tr = Tr()

        self.offset = self._sma.offset + self._tr.offset

    def next(self, h: float, l: float, c: float) -> float:
        return self._sma.next(self._tr.next(h, l, c))

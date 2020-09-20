import numba as nb
import numpy as np

from suchak.ma import MA
from suchak.util import jitclass


@jitclass
class SMA:
    offset: nb.int32
    period: nb.int32

    _ma: MA.class_type.instance_type

    def __init__(self, period: int):
        self.period = period

        self._ma = MA(period)
        self.offset = self._ma.offset

    def next(self, x: float) -> float:
        return np.sum(self._ma.next(x)) / self.period

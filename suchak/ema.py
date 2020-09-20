import numba as nb
import numpy as np

from suchak.sma import SMA
from suchak.util import jitclass


@jitclass
class EMA:
    offset: nb.int32
    period: nb.int32

    _ema1: nb.double
    _sma: SMA.class_type.instance_type
    _alpha: nb.double

    def __init__(self, period: int):
        self.period = period

        self._ema1 = np.nan
        self._sma = SMA(period)
        self._alpha = 2 / (period + 1)
        self.offset = self._sma.offset

    def next(self, x: float) -> float:
        if np.isnan(self._ema1):
            self._ema1 = self._sma.next(x)
        # self._ema1 = (x - self._ema1) * self._alpha + self._ema1
        self._ema1 = self._alpha * x + (1 - self._alpha) * self._ema1
        return self._ema1

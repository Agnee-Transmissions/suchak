import numba as nb
import numpy as np

from suchak.sma import SMA
from suchak.jitclass import jitclass


@jitclass
class EMA:
    offset: nb.int32
    period: nb.int32

    _ema: nb.double = np.nan
    _sma: SMA
    _alpha: nb.double = np.nan

    def __init__(self, period: int):
        self.period = period

        self._ema = np.nan
        self._sma = SMA(period)
        self._alpha = 2 / (period + 1)
        self.offset = self._sma.offset

    def next(self, x: float) -> float:
        if np.isnan(self._ema):
            self._ema = self._sma.next(x)
        self._ema = self._alpha * x + (1 - self._alpha) * self._ema
        return self._ema

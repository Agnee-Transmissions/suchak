import typing

import numba as nb
import numpy as np

from suchak.ema import EMA
from suchak.jitclass import jitclass
from suchak.sma import SMA


@jitclass
class MACD:
    offset: nb.int32
    short_period: nb.int32
    long_period: nb.int32
    signal_period: nb.int32

    _short_ema: EMA
    _long_ema: EMA
    _signal_sma: EMA

    def __init__(self, short_period: int, long_period: int, signal_period: int):
        self.short_period = short_period
        self.long_period = long_period
        self.signal_period = signal_period

        self._short_ema = EMA(short_period)
        self._long_ema = EMA(long_period)
        self._signal_sma = EMA(signal_period)

        self.offset = self._signal_sma.offset + max(
            self._short_ema.offset, self._long_ema.offset
        )

    def next(self, c: float) -> typing.Tuple[float, float, float]:
        macd = self._short_ema.next(c) - self._long_ema.next(c)
        signal = self._signal_sma.next(macd)
        hist = macd - signal
        return macd, signal, hist

    def next_arr(self, c_arr):
        out_len = len(c_arr)
        ret = np.empty((out_len, 3))
        for i in range(out_len):
            ret[i] = self.next(c_arr[i])
        return ret

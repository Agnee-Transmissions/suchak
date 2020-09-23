import typing

import numba as nb

from suchak.ema import EMA
from suchak.sma import SMA
from suchak.jitclass import jitclass


@jitclass
class MACD:
    offset: nb.int32
    short_period: nb.int32
    long_period: nb.int32
    signal_period: nb.int32

    _short_ema: EMA
    _long_ema: EMA
    _signal_sma: SMA

    def __init__(
        self, short_period: int = 12, long_period: int = 26, signal_period: int = 9
    ):
        self.short_period = short_period
        self.long_period = long_period
        self.signal_period = signal_period

        self._short_ema = EMA(short_period)
        self._long_ema = EMA(long_period)
        self._signal_sma = SMA(signal_period)

        self.offset = self._signal_sma.offset + max(
            self._short_ema.offset, self._long_ema.offset
        )

    def next(self, c: float) -> typing.Tuple[float, float, float]:
        macd = self._short_ema.next(c) - self._long_ema.next(c)
        signal = self._signal_sma.next(macd)
        hist = macd - signal
        return macd, signal, hist

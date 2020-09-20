import typing

import numba as nb

from suchak.ema import EMA
from suchak.sma import SMA
from suchak.util import jitclass


@jitclass
class MACD:
    offset: nb.int32
    short_period: nb.int32
    long_period: nb.int32
    signal_period: nb.int32

    _short_ema: EMA.class_type.instance_type
    _long_ema: EMA.class_type.instance_type
    _signal_sma: SMA.class_type.instance_type

    def __init__(
        self, short_period: int = 12, long_period: int = 26, signal_period: int = 9
    ):
        self.short_period = short_period
        self.long_period = long_period
        self.signal_period = signal_period

        self._short_ema = EMA(short_period)
        self._long_ema = EMA(long_period)
        self._signal_sma = EMA(signal_period)

        self.offset = self._signal_sma.offset + max(
            self._short_ema.offset, self._long_ema.offset
        )

    def next(self, c: float) -> typing.Tuple[float, float]:
        macd = self._short_ema.next(c) - self._long_ema.next(c)
        signal = self._signal_sma.next(c)
        return macd, signal

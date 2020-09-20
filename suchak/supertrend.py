import typing

import numba as nb
import numpy as np

from suchak.atr import ATR
from suchak.util import jitclass


@jitclass
class Supertrend:
    offset: nb.int32
    period: nb.int32
    factor: nb.double

    _atr: ATR.class_type.instance_type

    _dt: nb.double
    _up1: nb.double
    _dn1: nb.double
    _c1: nb.double

    def __init__(self, period: int, factor: float):
        self.period = period
        self.factor = factor

        self._atr = ATR(period)
        self.offset = self._atr.offset

        self._dt = 1
        self._up1 = np.nan
        self._dn1 = np.nan
        self._c1 = np.nan

    def next(self, h: float, l: float, c: float) -> typing.Tuple[float, float]:
        f_atr = self.factor * self._atr.next(h, l, c)

        hl2 = (h + l) / 2

        up = hl2 - f_atr
        dn = hl2 + f_atr

        if self._c1 > self._up1 > up:
            up = self._up1
        if self._c1 < self._dn1 < dn:
            dn = self._dn1

        if (self._dt < 0 and self._dn1 < c) or (self._dt > 0 and self._up1 > c):
            self._dt *= -1

        if self._dt > 0:
            st = up
        else:
            st = dn

        self._up1 = up
        self._dn1 = dn
        self._c1 = c

        return st, self._dt

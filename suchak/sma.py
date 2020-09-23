import numba as nb
import numpy as np

from suchak.jitclass import jitclass
from suchak.window import Window


@jitclass
class SMA:
    offset: nb.int32
    period: nb.int32

    _win: Window

    def __init__(self, period: int):
        self.period = period

        self._win = Window(period)
        self.offset = self._win.offset

    def next(self, x: float) -> float:
        return np.sum(self._win.next(x)) / self.period

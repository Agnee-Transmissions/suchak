import typing

import numpy as np
from numba.core import types

from suchak import Window, Deque, SMA
from suchak.jitclass import jitclass


@jitclass
class ZigZagPivots:
    bar_length: types.int32
    length: types.int32
    offset: types.int32

    _src_deque: Deque
    _shmma_sma: SMA
    _shmma_win: Window

    _top: types.float64 = np.nan
    _bot: types.float64 = np.nan

    def __init__(self, length: int, bar_length: int):
        self.length = length
        self.bar_length = bar_length

        self._src_deque = Deque(length)

        self._shmma_sma = SMA(length)
        self._shmma_win = Window(bar_length)

        self.offset = max(
            self._src_deque.offset, self._shmma_sma.offset, self._shmma_win.offset
        )

    def next(self, src: float) -> typing.Tuple[float, float]:
        self._src_deque.append_left(src)
        src_arr = self._src_deque.array()

        slope = 0.0
        for i in range(0, self.length):
            factor = 1 + 2 * i
            slope += src_arr[i] * (self.length - factor) / 2

        shmma_sma = self._shmma_sma.next(src)
        shmma = shmma_sma + (6 * slope) / ((self.length + 1) * self.length)

        # backup previous values
        top1 = self._top
        bot1 = self._bot

        shmma_buf = self._shmma_win.next(shmma)
        if shmma >= np.max(shmma_buf):
            self._top = shmma
        if shmma <= np.min(shmma_buf):
            self._bot = shmma

        # only return if values didn't changed
        if self._top != top1:
            top = np.nan
        else:
            top = self._top
        if self._bot != bot1:
            bot = np.nan
        else:
            bot = self._bot
        return top, bot

    def next_arr(self, src_arr):
        out_len = len(src_arr)
        ret = np.empty((out_len, 2))
        for i in range(out_len):
            ret[i] = self.next(src_arr[i])
        return ret

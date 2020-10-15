import numba as nb
import numpy as np
import typing

from suchak import Deque
from suchak.jitclass import jitclass


@jitclass
class Fibo:
    offset: nb.int32

    period: nb.int32

    _h_win: Deque
    _l_win: Deque

    def __init__(self, period: int):
        self.offset = period - 1
        self.period = period

        self._h_win = Deque(period)
        self._l_win = Deque(period)

    def next(
        self, h: float, l: float
    ) -> typing.Tuple[
        float, float, float, float, float, float, float,
    ]:
        self._h_win.append_left(h)
        self._l_win.append_left(l)

        h_arr = self._h_win.array()
        l_arr = self._l_win.array()

        h_max_idx = np.argmax(h_arr)
        l_min_idx = np.argmin(l_arr)

        h1 = h_arr[h_max_idx]
        l1 = l_arr[l_min_idx]
        fark = h1 - l1

        fark236 = fark * 0.236
        fark382 = fark * 0.382
        fark500 = fark * 0.500
        fark618 = fark * 0.618
        fark786 = fark * 0.786

        hl236 = l1 + fark236
        hl382 = l1 + fark382
        hl500 = l1 + fark500
        hl618 = l1 + fark618
        hl786 = l1 + fark786

        lh236 = h1 - fark236
        lh382 = h1 - fark382
        lh500 = h1 - fark500
        lh618 = h1 - fark618
        lh786 = h1 - fark786

        # pick the farthest one from last bar
        cond = h_max_idx > l_min_idx
        f236 = hl236 if cond else lh236
        f382 = hl382 if cond else lh382
        f500 = hl500 if cond else lh500
        f618 = hl618 if cond else lh618
        f786 = hl786 if cond else lh786

        return l1, h1, f236, f382, f500, f618, f786

    def next_arr(self, h_arr, l_arr):
        out_len = len(h_arr)
        ret = np.empty((out_len, 7))
        for i in range(out_len):
            ret[i] = self.next(h_arr[i], l_arr[i])
        return ret

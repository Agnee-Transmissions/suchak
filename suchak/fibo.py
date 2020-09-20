import numba as nb
import numpy as np

from suchak.util import jitclass


@jitclass
class Fibo:
    offset: nb.int32

    period: nb.int32

    _h_buf: nb.double[:]
    _h_idx: nb.int32

    _l_buf: nb.double[:]
    _l_idx: nb.int32

    def __init__(self, period: int = 144):
        self.offset = period - 1
        self.period = period

        self._h_buf = np.empty(period)
        self._h_buf[:] = np.nan
        self._h_idx = 0

        self._l_buf = np.empty(period)
        self._l_buf[:] = np.nan
        self._l_idx = 0

    def next(self, h: float, l: float) -> np.ndarray:
        self._h_buf[self._h_idx] = h
        self._h_idx = (self._h_idx + 1) % self.period

        self._l_buf[self._l_idx] = l
        self._l_idx = (self._l_idx + 1) % self.period

        h1 = np.max(self._h_buf)
        l1 = np.min(self._l_buf)
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

        hbars = np.argmax(self._h_buf)
        lbars = np.argmin(self._l_buf)

        # pick the farthest one from last (lower idx)
        cond = hbars > lbars
        f236 = hl236 if cond else lh236
        f382 = hl382 if cond else lh382
        f500 = hl500 if cond else lh500
        f618 = hl618 if cond else lh618
        f786 = hl786 if cond else lh786

        return np.array([l1, h1, f236, f382, f500, f618, f786])

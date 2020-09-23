import numba as nb
import numpy as np

from suchak.jitclass import jitclass


@jitclass
class Deque:
    offset: nb.int32
    maxlen: nb.int32

    _buf: nb.double[:]
    _idx0: nb.int32
    len: nb.int32 = 0

    def __init__(self, maxlen: int):
        self.offset = maxlen - 1
        self.maxlen = maxlen

        self._buf = np.empty(maxlen * 4)
        self._idx0 = len(self._buf) // 2

    def append(self, x: float):
        real_len = self._idx0 + self.len

        if real_len >= len(self._buf):
            self._move()

            # need to re-compute this
            real_len = self._idx0 + self.len

        self._buf[real_len] = x

        if self.len < self.maxlen:
            self.len += 1
        else:
            self._idx0 += 1

    def append_left(self, x: float):
        if self._idx0 <= 0:
            self._move()

        self._buf[self._idx0 - 1] = x

        self._idx0 -= 1
        if self.len < self.maxlen:
            self.len += 1

    def _move(self):
        arr = self._buf[self._idx0 : self._idx0 + self.len]
        self._idx0 = len(self._buf) // 2
        self._buf[self._idx0 : self._idx0 + self.len] = arr

    def array(self) -> np.ndarray:
        return self._buf[self._idx0 : self._idx0 + self.len]

    def __getitem__(self, idx: int) -> float:
        return self.array()[idx]
        # return self._buf[(self._idx0 + idx) % self.maxlen]

    def __setitem__(self, idx: int, item: float):
        self.array()[idx] = item
        # self._buf[(self._idx0 + idx) % self.maxlen] = item

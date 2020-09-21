import numba as nb
import numpy as np

from suchak.util import jitclass


@jitclass
class Deque:
    offset: nb.int32
    length: nb.int32

    _buf: nb.double[:]
    _idx0: nb.int32
    _len: nb.int32

    def __init__(self, length: int):
        self.offset = length - 1
        self.length = length

        self._buf = np.empty(length * 4)
        self._idx0 = len(self._buf) // 2
        self._len = 0

    def append(self, x: float):
        real_len = self._idx0 + self._len

        if real_len >= len(self._buf):
            self._move()

            # need to re-compute this
            real_len = self._idx0 + self._len

        self._buf[real_len] = x

        if self._len < self.length:
            self._len += 1
        else:
            self._idx0 += 1

    def append_left(self, x: float):
        if self._idx0 <= 0:
            self._move()

        self._buf[self._idx0 - 1] = x

        self._idx0 -= 1
        if self._len < self.length:
            self._len += 1

    def _move(self):
        arr = self._buf[self._idx0 : self._idx0 + self._len]
        self._idx0 = len(self._buf) // 2
        self._buf[self._idx0 : self._idx0 + self._len] = arr

    def array(self) -> np.ndarray:
        return self._buf[self._idx0 : self._idx0 + self._len]

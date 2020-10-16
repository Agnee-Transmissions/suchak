import typing

import numpy as np
from numba import typeof, njit, types


def vec_type(dtype):
    return typeof(np.empty(0, dtype=dtype))
    # return types.Tuple([typeof(np.empty(0, dtype=dtype)), types.int32])


@njit
def vec_new(dtype, capacity=0) -> typing.Tuple[np.ndarray, int]:
    return np.empty(capacity, dtype=dtype), -1


@njit
def vec_array(buf: np.ndarray, idx: int) -> np.ndarray:
    return buf[: idx + 1]


@njit
def vec_next(buf: np.ndarray, idx: int) -> typing.Tuple[np.ndarray, int]:
    idx += 1

    # perform array-doubling if needed
    if idx >= len(buf):
        old = buf
        new = np.empty(len(old) * 2, dtype=old.dtype)
        new[: len(old)] = old
        buf = new

    return buf, idx


@njit
def vec_append(buf: np.ndarray, idx: int, value) -> typing.Tuple[np.ndarray, int]:
    buf, idx = vec_next(buf, idx)
    buf[idx] = value
    return buf, idx


@njit
def vec_next_left(buf: np.ndarray, idx: int) -> typing.Tuple[np.ndarray, int]:
    idx -= 1

    # perform array-doubling if needed
    if idx >= len(buf):
        old = buf
        new = np.empty(len(old) * 2, dtype=old.dtype)
        new[: len(old)] = old
        buf = new

    return buf, idx


@njit
def vec_append_left(buf: np.ndarray, idx: int, value) -> typing.Tuple[np.ndarray, int]:
    buf, idx = vec_next_left(buf, idx)
    buf[idx] = value
    return buf, idx

from numba import njit

from suchak.ema import EMA


@njit
def RMA(period: int) -> EMA:
    rma = EMA(period)
    rma._alpha = 1 / period
    return rma

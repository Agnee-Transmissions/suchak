import numba
import numpy as np
import tulipy as ta
from numba import njit
from pytest import approx

from suchak.atr import ATR
from suchak.fibo import Fibo
from suchak.jitclass import jitclass
from suchak.sma import SMA
from suchak.supertrend import Supertrend
from suchak.tr import TR

SIZE = 64


def test_jitclass():
    @jitclass
    class Foo:
        x: numba.int32 = 42
        y: numba.double

        def __init__(self):
            self.y = 4.2

    @njit
    def f():
        foo = Foo()
        return foo.x, foo.y

    assert f() == (42, 4.2)


def test_sma():
    c = np.random.random(SIZE)
    length = 14

    sma = SMA(length)

    for i in range(sma.offset):
        nan = sma.next(c[i])
        assert np.isnan(nan)

    computed = ta.sma(c, length)

    for i in range(len(computed)):
        actual = sma.next(c[i + sma.offset])
        expected = computed[i]

        assert actual == approx(expected)


def test_tr():
    h, l, c = np.random.random((3, SIZE))

    tr = TR()

    for i in range(tr.offset):
        nan = tr.next(h[i], l[i], c[i])
        assert np.isnan(nan)

    computed = ta.tr(h, l, c)

    for i in range(len(computed)):
        actual = tr.next(h[i + tr.offset], l[i + tr.offset], c[i + tr.offset])
        expected = computed[i]

        assert actual == approx(expected)


def test_atr():
    h, l, c = np.random.random((3, SIZE))
    period = 5

    atr = ATR(period)

    for i in range(atr.offset):
        nan = atr.next(h[i], l[i], c[i])
        assert np.isnan(nan)

    computed = ta.sma(ta.tr(h, l, c), period)

    for i in range(len(computed)):
        actual = atr.next(h[i + atr.offset], l[i + atr.offset], c[i + atr.offset])
        expected = computed[i]

        assert actual == approx(expected)


def test_supertrend():
    h, l, c = np.random.random((3, SIZE))
    period = 5
    factor = 30

    supertrend = Supertrend(period, factor)

    for i in range(supertrend.offset):
        st, _ = supertrend.next(h[i], l[i], c[i])
        assert np.isnan(st)

    computed = ti_supertrend(h, l, c, period, factor)

    for i in range(len(computed)):
        ast, adt = supertrend.next(
            h[i + supertrend.offset], l[i + supertrend.offset], c[i + supertrend.offset]
        )
        est, edt = computed[i]

        assert adt == edt
        assert ast == approx(est)


def test_supertrend_arr():
    h, l, c = np.random.random((3, SIZE))
    period = 5
    factor = 30

    supertrend = Supertrend(period, factor)
    actual = supertrend.next_arr(h, l, c)
    computed = ti_supertrend(h, l, c, period, factor)
    actual = actual[-len(computed) :]

    assert np.all(actual == approx(computed))


def test_fibo():
    h, l = np.random.random((2, SIZE))
    period = 32

    fibo = Fibo(period)

    for i in range(SIZE):
        actual = fibo.next(h[i], l[i])

    expected = ti_fibo(h, l, period)
    assert np.all(actual == approx(expected))


# vectorized implementation for cross-verification
def ti_supertrend(
    h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int, factor: float,
) -> np.ndarray:

    f_atr = factor * ta.sma(ta.tr(h, l, c), period)

    out_size = len(f_atr)
    slicer = slice(-out_size, None)

    # IMPORTANT!
    h, l, c = h[slicer], l[slicer], c[slicer]

    hl2 = (h + l) / 2

    up = hl2 - f_atr
    dn = hl2 + f_atr

    st = np.empty(out_size)
    dt = np.empty(out_size)
    st[0] = up[0]
    dt[0] = 1

    for i in range(1, len(st)):
        up1 = up[i - 1]
        dn1 = dn[i - 1]
        dt1 = dt[i - 1]
        c1 = c[i - 1]

        if c1 > up1 > up[i]:
            up[i] = up1
        if c1 < dn1 < dn[i]:
            dn[i] = dn1

        if (dt1 < 0 and dn1 < c[i]) or (dt1 > 0 and up1 > c[i]):
            dt1 *= -1
        dt[i] = dt1

        if dt1 > 0:
            st[i] = up[i]
        else:
            st[i] = dn[i]

    ret = np.empty((out_size, 2))
    ret[:, 0] = st
    ret[:, 1] = dt
    return ret


# vectorized implementation for cross-verification
def ti_fibo(h: np.ndarray, l: np.ndarray, length: int) -> np.ndarray:
    h, l = h[-length:], l[-length:]

    l1 = np.min(l)
    h1 = np.max(h)
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

    hbars = np.argmax(h)
    lbars = np.argmin(l)

    # pick the farthest one from last (lower idx)
    cond = hbars > lbars
    f236 = hl236 if cond else lh236
    f382 = hl382 if cond else lh382
    f500 = hl500 if cond else lh500
    f618 = hl618 if cond else lh618
    f786 = hl786 if cond else lh786

    return np.array([l1, h1, f236, f382, f500, f618, f786])

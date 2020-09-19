import numpy as np
import tulipy as ta
from pytest import approx

from suchak.atr import Atr
from suchak.sma import Sma
from suchak.supertrend import Supertrend
from suchak.tr import Tr

SIZE = 64


def test_sma():
    c = np.random.random(SIZE)
    length = 14

    sma = Sma(length)

    for i in range(sma.offset):
        sma.next(c[i])

    computed = ta.sma(c, length)

    for i in range(len(computed)):
        actual = sma.next(c[i + sma.offset])
        expected = computed[i]

        assert actual == approx(expected)


def test_tr():
    h, l, c = np.random.random((3, SIZE))

    tr = Tr()

    for i in range(tr.offset):
        tr.next(h[i], l[i], c[i])

    computed = ta.tr(h, l, c)

    for i in range(len(computed)):
        actual = tr.next(h[i + tr.offset], l[i + tr.offset], c[i + tr.offset])
        expected = computed[i]

        assert actual == approx(expected)


def test_atr():
    h, l, c = np.random.random((3, SIZE))
    period = 5

    atr = Atr(period)

    for i in range(atr.offset):
        atr.next(h[i], l[i], c[i])

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
        supertrend.next(h[i], l[i], c[i])

    computed = ti_supertrend(h, l, c, period, factor)

    for i in range(len(computed)):
        ast, adt = supertrend.next(
            h[i + supertrend.offset], l[i + supertrend.offset], c[i + supertrend.offset]
        )
        est, edt = computed[i]

        assert adt == edt
        assert ast == approx(est)


# vectorized supertrend (for testing)
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

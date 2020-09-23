Suchak is a technical indicator library that supports incremental computation / streaming API.

In live-trading, this makes it faster than comparable libraries written in C.

suchak is written in pure python using a magical piece of software - [numba](https://github.com/numba/numba).

---

The API for all indicators is consistent & simple -

```python
c = np.random.random(SIZE)
length = 14

sma = SMA(length)

for ci in c:
    smai = sma.next(ci)
    print('close:', ci, 'sma:', smai)
```

---

suchak's goal is to empower you to use its indicators,
and still be able to write complex strategies, entirely in python.

This is possible because suchak can be called from inside numba's `@njit` functions

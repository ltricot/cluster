from numba import njit
import numpy.typing as npt
import numpy as np

from ..metrics import Metric


@njit
def _random_choice(xs: npt.NDArray, w: npt.NDArray) -> npt.NDArray:
    at = np.random.rand() * w.sum()

    for i in range(xs.shape[0]):
        at -= w[i]
        if at <= 0:
            return xs[i]

    return xs[-1]


def random(xs: npt.NDArray, k: int):
    _, d = xs.shape

    labels = np.random.randint(0, k, xs.shape[0])

    ys = np.empty((k, d), dtype=xs.dtype)
    for j in range(k):
        ys[j] = np.mean(xs[labels == j], axis=0)

    return ys


@njit
def kmeanspp(xs: npt.NDArray, k: int, metric: Metric, sq: bool = True) -> npt.NDArray:
    n, p = xs.shape

    ys = np.empty((k, p), dtype=xs.dtype)
    ys[0] = xs[np.random.randint(n)]

    prob = np.empty(n, dtype=np.float64)
    for l in range(1, k):
        for i in range(n):
            m = np.inf
            for j in range(l):
                m = min(m, metric(xs[i], ys[j]))
            if sq:
                m = m ** 2
            prob[i] = m

        ys[l] = _random_choice(xs, prob)

    return ys


@njit
def mckmeanspp(
    xs: npt.NDArray, k: int, metric: Metric, sample: int, sq: bool = True
) -> npt.NDArray:
    n, p = xs.shape

    ys = np.empty((k, p), dtype=xs.dtype)
    ys[0] = xs[np.random.randint(n)]

    prob = np.empty(sample, dtype=np.float64)
    _xs = np.empty((sample, p), dtype=xs.dtype)

    for l in range(1, k):
        for s in range(sample):
            i = np.random.randint(n)
            _xs[s] = xs[i]

            m = np.inf
            for j in range(l):
                m = min(m, metric(xs[i], ys[j]))
            if sq:
                m = m ** 2
            prob[s] = m

        ys[l] = _random_choice(_xs, prob)

    return ys

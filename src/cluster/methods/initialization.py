from numba import njit
import numpy.typing as npt
import numpy as np

from .._settings import PARALLEL
from ..metrics import Metric


@njit
def _random_choice(w: npt.NDArray) -> np.int64:
    at = np.random.rand() * w.sum()

    for i in range(w.shape[0]):
        at -= w[i]
        if at <= 0:
            return np.int64(i)

    return np.int64(len(w) - 1)


@njit
def _random_choice_binsearch(wcs: npt.NDArray) -> np.int64:
    return min(np.searchsorted(wcs, np.random.rand()), np.int64(len(wcs) - 1))


@njit
def _dist_min(x: npt.NDArray, ys: npt.NDArray, metric: Metric):
    k, _ = ys.shape

    dx = np.inf
    for j in range(k):
        dx = min(dx, metric(x, ys[j]))

    return dx


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
            dx = _dist_min(xs[i], ys[:l], metric)
            if sq:
                dx = dx ** 2
            prob[i] = dx

        ys[l] = xs[_random_choice(prob)]

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

        ys[l] = _xs[_random_choice(prob)]

    return ys


# this algorithms seems to be no better than my MC kmeans++
# despite the theoretical guarantees ; it is also slower


@njit(parallel=PARALLEL)
def afkmcmc(
    xs: npt.NDArray, k: int, metric: Metric, sample: int, sq: bool = True
) -> npt.NDArray:
    n, p = xs.shape

    ys = np.empty((k, p), dtype=xs.dtype)
    ys[0] = xs[np.random.randint(n)]

    q = np.empty(n, dtype=np.float64)
    for i in range(n):
        q[i] = metric(xs[i], ys[0])
        if sq:
            q[i] = q[i] ** 2

    q = 1 / (2 * n) + 1 / 2 / q.sum() * q
    qcs = np.cumsum(q)

    for l in range(1, k):
        xi = _random_choice_binsearch(qcs)
        dx = _dist_min(xs[xi], ys[:l], metric)
        if sq:
            dx = dx ** 2

        for _ in range(sample):
            yi = _random_choice_binsearch(qcs)
            dy = _dist_min(xs[yi], ys[:l], metric)
            if sq:
                dy = dy ** 2

            if dx == 0 or dy * q[xi] / dx / q[yi] > np.random.rand():
                xi, dx = yi, dy

        ys[l] = xs[xi]

    return ys

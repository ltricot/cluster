import numpy.typing as npt
import numpy as np
from numba import njit, prange  # type: ignore

from typing import Protocol


class Metric(Protocol):
    def __call__(self, x: npt.NDArray, y: npt.NDArray) -> float:
        ...


def _label_default(
    xs: npt.NDArray, ys: npt.NDArray, metric: Metric
) -> tuple[npt.NDArray[np.int64], np.float64]:
    n, _ = xs.shape
    m, _ = ys.shape

    labels = np.empty(n, dtype=np.int64)
    cost = np.float64(0.0)

    for i in prange(n):
        mj, md = 0, np.inf
        for j in range(m):
            d = metric(xs[i], ys[j])
            if d < md:
                mj, md = j, d

        labels[i] = mj
        cost += md ** 2

    return labels, cost


_label = njit(_label_default)
_label_parallel = njit(parallel=True)(_label_default)


@njit
def _random_choice(xs: npt.NDArray, w: npt.NDArray) -> npt.NDArray:
    at = np.random.rand() * w.sum()

    for i in range(xs.shape[0]):
        at -= w[i]
        if at <= 0:
            return xs[i]

    return xs[-1]


@njit
def _kmeanspp(xs: npt.NDArray, k: int, metric: Metric) -> npt.NDArray:
    n, p = xs.shape

    ys = np.empty((k, p), dtype=xs.dtype)
    ys[0] = xs[np.random.randint(n)]

    prob = np.empty(n, dtype=np.float64)
    for l in range(1, k):
        for i in range(n):
            m = np.inf
            for j in range(l):
                m = min(m, metric(xs[i], ys[j]))
            prob[i] = m ** 2

        ys[l] = _random_choice(xs, prob)

    return ys


@njit
def _kmeanspp_approx(
    xs: npt.NDArray, k: int, metric: Metric, sample: int
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
            prob[s] = m ** 2

        ys[l] = _random_choice(_xs, prob)

    return ys


@njit
def _gonzalez(xs: npt.NDArray, k: int, metric: Metric) -> npt.NDArray:
    n, p = xs.shape

    ys = np.empty((k, p), dtype=xs.dtype)
    ys[0] = xs[np.random.randint(n)]

    for l in range(1, k):
        mi, md = 0, 0.0

        for i in range(n):
            m = np.inf
            for j in range(l):
                m = min(m, metric(xs[i], ys[j]))

            if m > md:
                mi, md = i, m

        ys[l] = xs[mi]

    return ys


def _lloyd_exact(
    xs: npt.NDArray, ys: npt.NDArray, metric: Metric, maxiter: int, parallel=True
) -> tuple[npt.NDArray, bool]:
    label = _label
    if parallel:
        label = _label_parallel

    m, _ = ys.shape
    ys = np.copy(ys)

    labels, _ = label(xs, ys, metric)
    old = labels

    for _ in range(maxiter):
        for j in range(m):
            ys[j] = np.mean(xs[labels == j], axis=0)

        old[:] = labels
        labels, _ = label(xs, ys, metric)

        if np.array_equal(labels, old):
            return ys, True

    return ys, False


_eps = np.finfo(np.float64).eps


def _lloyd_approx(
    xs: npt.NDArray,
    ys: npt.NDArray,
    metric: Metric,
    maxiter: int,
    rtol: np.float64,
    parallel=True,
) -> tuple[npt.NDArray, bool]:
    label = _label
    if parallel:
        label = _label_parallel

    m, _ = ys.shape
    ys = np.copy(ys)

    labels, cost = label(xs, ys, metric)

    for _ in range(maxiter):
        for j in range(m):
            ys[j] = np.mean(xs[labels == j], axis=0)

        old = cost
        labels, cost = label(xs, ys, metric)

        if np.abs(cost - old) / (old + _eps) < rtol:
            return ys, True

    return ys, False

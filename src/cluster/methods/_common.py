from numba import njit, prange
import numpy.typing as npt
import numpy as np

from ..metrics import Metric


@njit(parallel=True)
def _init_labels(xs: npt.NDArray, ys: npt.NDArray, metric: Metric):
    n, _ = xs.shape
    k, _ = ys.shape

    labels = np.empty(n, dtype=np.int64)

    for i in prange(n):
        mj, md = 0, np.inf
        for j in range(k):
            d = metric(xs[i], ys[j])
            if d < md:
                mj, md = j, d

        labels[i] = mj

    return labels


@njit
def _init_ys(xs: npt.NDArray, k: int, labels: npt.NDArray):
    n, d = xs.shape

    _ys = np.zeros((k, d), dtype=xs.dtype)
    _ns = np.zeros(k, dtype=np.int64)

    for i in range(n):
        _ys[labels[i]] += xs[i]
        _ns[labels[i]] += 1

    return _ys, _ns


@njit
def _update_ys(
    xs: npt.NDArray,
    _ys: npt.NDArray,
    _ns: npt.NDArray,
    move: npt.NDArray,
    labels: npt.NDArray,
) -> bool:
    n, _ = xs.shape

    chg = False
    for i in range(n):
        ni, li = move[i], labels[i]

        if ni != li:
            _ys[ni], _ns[ni] = _ys[ni] + xs[i], _ns[ni] + 1
            _ys[li], _ns[li] = _ys[li] - xs[i], _ns[li] - 1
            chg = True

    return chg


@njit(parallel=True)
def _pdist(ys: npt.NDArray, metric: Metric, out: npt.NDArray):
    k, _ = ys.shape

    for i in prange(k):
        for j in range(i + 1, k):
            out[i, j] = metric(ys[i], ys[j])
            out[j, i] = out[i, j]

from numba import njit, prange
import numpy.typing as npt
import numpy as np

from ..metrics import Metric
from .._settings import CACHE
from ._common import _init_labels, _init_ys, _update_ys


@njit(cache=CACHE, parallel=True)
def _pdist_sort(ys: npt.NDArray, metric: Metric, pdist: npt.NDArray, sort: npt.NDArray):
    k, _ = ys.shape

    for i in prange(k):
        for j in range(i + 1, k):
            pdist[i, j] = metric(ys[i], ys[j])
            pdist[j, i] = pdist[i, j]

    for i in prange(k):
        sort[i] = np.argsort(pdist[i])


@njit(cache=CACHE, parallel=True)
def _label_sort(
    xs: npt.NDArray,
    ys: npt.NDArray,
    metric: Metric,
    pdist: npt.NDArray,
    sort: npt.NDArray,
    labels: npt.NDArray,
):
    n, _ = xs.shape

    for i in prange(n):
        li, ui = labels[i], metric(xs[i], ys[labels[i]])
        mj, md = 0, np.inf

        for j in sort[li]:
            if 2 * ui < pdist[li, j]:
                break

            d = metric(xs[i], ys[j])
            if d < md:
                mj, md = j, d

        labels[i] = mj


@njit(cache=CACHE)
def sort(xs: npt.NDArray, ys: npt.NDArray, metric: Metric, maxiter: int):
    n, _ = xs.shape
    k, _ = ys.shape

    # label
    move = _init_labels(xs, ys, metric)
    labels = np.empty(n, dtype=np.int64)
    pdist = np.zeros((k, k), dtype=xs.dtype)
    sort = np.empty((k, k), dtype=np.int64)

    # move
    _ys, _ns = _init_ys(xs, k, move)
    ys = _ys / _ns.reshape(-1, 1)

    for _ in range(maxiter):
        labels[:] = move
        _pdist_sort(ys, metric, pdist, sort)
        _label_sort(xs, ys, metric, pdist, sort, move)

        if not _update_ys(xs, _ys, _ns, move, labels):
            return ys, labels, True

        ys[:] = _ys / _ns.reshape(-1, 1)

    return ys, labels, False

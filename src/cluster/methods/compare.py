from numba import njit, prange
import numpy.typing as npt
import numpy as np

from ..metrics import Metric
from .._settings import CACHE
from ._common import _init_labels, _init_ys, _update_ys, _pdist


@njit(cache=CACHE, parallel=True)
def _label_compare(
    xs: npt.NDArray,
    ys: npt.NDArray,
    metric: Metric,
    pdist: npt.NDArray,
    labels: npt.NDArray,
):
    n, _ = xs.shape
    k, _ = ys.shape

    for i in prange(n):
        li, ui = labels[i], metric(xs[i], ys[labels[i]])
        mj, md = 0, np.inf

        for j in range(k):
            if 2 * ui < pdist[li, j]:
                continue

            d = metric(xs[i], ys[j])
            if d < md:
                mj, md = j, d

        labels[i] = mj


@njit(cache=CACHE)
def compare(xs: npt.NDArray, ys: npt.NDArray, metric: Metric, maxiter: int):
    n, _ = xs.shape
    k, _ = ys.shape

    # label
    move = _init_labels(xs, ys, metric)
    labels = np.empty(n, dtype=np.int64)
    pdist = np.zeros((k, k), dtype=xs.dtype)

    # move
    _ys, _ns = _init_ys(xs, k, move)
    ys = _ys / _ns.reshape(-1, 1)

    for _ in range(maxiter):
        labels[:] = move
        _pdist(ys, metric, pdist)
        _label_compare(xs, ys, metric, pdist, move)

        if not _update_ys(xs, _ys, _ns, move, labels):
            break

        ys[:] = _ys / _ns.reshape(-1, 1)

    return labels

from numba import njit, prange

import numpy.typing as npt
import numpy as np

from ..metrics import Metric
from .._settings import CACHE, PARALLEL
from ._common import _init_ys, _update_ys


# I hope I made a mistake because otherwise this algorithm SUCKS


@njit(cache=CACHE, parallel=PARALLEL)
def _pdist_elkan(ys: npt.NDArray, metric: Metric, pdist: npt.NDArray, s: npt.NDArray):
    k, _ = ys.shape

    for i in prange(k):
        md = np.inf
        for j in range(k):
            if i != j:
                pdist[i, j] = metric(ys[i], ys[j])
                pdist[j, i] = pdist[i, j]
                if pdist[i, j] < md:
                    md = pdist[i, j]

        s[i] = md


@njit(cache=CACHE, parallel=PARALLEL)
def _init_labels_elkan(xs: npt.NDArray, ys: npt.NDArray, metric: Metric):
    n, _ = xs.shape
    k, _ = ys.shape

    labels = np.empty(n, dtype=np.int64)
    upper = np.empty(n, dtype=xs.dtype)
    lower = np.empty((n, k), dtype=xs.dtype)

    for i in prange(n):
        mj, md = 0, np.inf
        for j in range(k):
            d = metric(xs[i], ys[j])
            lower[i, j] = d

            if d < md:
                mj, md = j, d

        labels[i] = mj
        upper[i] = md

    return labels, upper, lower


@njit(cache=CACHE, parallel=PARALLEL)
def _label_elkan(
    xs: npt.NDArray,
    ys: npt.NDArray,
    metric: Metric,
    labels: npt.NDArray,
    upper: npt.NDArray,
    lower: npt.NDArray,
    pdist: npt.NDArray,
    s: npt.NDArray,
):
    n, _ = xs.shape
    k, _ = ys.shape

    for i in prange(n):
        li, ui = labels[i], upper[i]

        if 2 * ui <= s[li]:
            continue

        mj, md = 0, np.inf
        for j in range(k):
            if ui <= lower[i, j] or 2 * ui <= pdist[li, j]:
                continue

            d = metric(xs[i], ys[j])
            if d < md:
                mj, md = j, d

            lower[i, j] = d

        labels[i] = mj
        upper[i] = md


@njit(cache=CACHE, parallel=PARALLEL)
def _update_bounds(
    dys: npt.NDArray,
    labels: npt.NDArray,
    upper: npt.NDArray,
    lower: npt.NDArray,
    metric: Metric,
):
    (n,) = upper.shape
    k, d = dys.shape

    zero = np.zeros(d, dtype=dys.dtype)
    norm = np.empty(k, dtype=dys.dtype)

    for j in prange(k):
        norm[j] = metric(dys[j], zero)

    for i in prange(n):
        upper[i] += norm[labels[i]]

        for j in range(k):
            lower[i] -= norm[j]


@njit(cache=CACHE)
def elkan(xs: npt.NDArray, ys: npt.NDArray, metric: Metric, maxiter: int):
    n, _ = xs.shape
    k, _ = ys.shape

    # label
    move, upper, lower = _init_labels_elkan(xs, ys, metric)
    labels = np.empty(n, dtype=np.int64)
    pdist = np.zeros((k, k), dtype=xs.dtype)
    s = np.empty(k, dtype=xs.dtype)

    # move
    _ys, _ns = _init_ys(xs, k, move)
    _ys_cp, _ns_cp = _ys.copy(), _ns.copy()
    dys = (_ys / _ns.reshape(-1, 1)) - ys
    _update_bounds(dys, move, upper, lower, metric)
    ys = _ys / _ns.reshape(-1, 1)

    for _ in range(maxiter):
        labels[:] = move
        _pdist_elkan(ys, metric, pdist, s)
        _label_elkan(xs, ys, metric, move, upper, lower, pdist, s)

        if not _update_ys(xs, _ys_cp, _ns_cp, move, labels):
            return ys, labels, True

        dys = (_ys_cp / _ns_cp.reshape(-1, 1)) - ys
        _update_bounds(dys, move, upper, lower, metric)

        _ys[:] = _ys_cp
        _ns[:] = _ns_cp
        ys += dys

    return ys, labels, False

from numba import njit, prange

import numpy.typing as npt
import numpy as np

from ..metrics import Metric
from .._settings import CACHE, PARALLEL
from ._common import _init_ys, _update_ys


@njit(cache=CACHE, parallel=PARALLEL)
def _pdist_hamerly2(
    ys: npt.NDArray, metric: Metric, pdist: npt.NDArray, s: npt.NDArray
):
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
def _init_labels_hamerly2(xs: npt.NDArray, ys: npt.NDArray, metric: Metric):
    n, _ = xs.shape
    k, _ = ys.shape

    labels = np.empty(n, dtype=np.int64)
    upper = np.empty(n, dtype=xs.dtype)
    lower = np.empty(n, dtype=xs.dtype)

    for i in prange(n):
        mj, md, lmd = 0, np.inf, np.inf
        for j in range(k):
            d = metric(xs[i], ys[j])
            if d < md:
                mj, md, lmd = j, d, md
            elif d < lmd:
                lmd = d

        labels[i] = mj
        upper[i] = md
        lower[i] = lmd

    return labels, upper, lower


@njit(cache=CACHE, parallel=PARALLEL)
def _label_hamerly2(
    xs: npt.NDArray,
    ys: npt.NDArray,
    metric: Metric,
    labels: npt.NDArray,
    upper: npt.NDArray,
    lower: npt.NDArray,
    s: npt.NDArray,
):
    n, _ = xs.shape
    k, _ = ys.shape

    for i in prange(n):
        li = labels[i]

        m = max(lower[i], s[li] / 2)
        if upper[i] < m:
            continue

        upper[i] = metric(xs[i], ys[li])
        if upper[i] < m:
            continue

        mj, md, lmd = 0, np.inf, np.inf
        for j in range(k):
            d = metric(xs[i], ys[j])
            if d < md:
                mj, md, lmd = j, d, md
            elif d < lmd:
                lmd = d

        labels[i] = mj
        upper[i] = md
        lower[i] = lmd


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

    delta = 0.0
    for j in prange(k):
        norm[j] = metric(dys[j], zero)
        delta = max(delta, norm[j])

    for i in prange(n):
        upper[i] += norm[labels[i]]
        lower[i] -= delta


@njit(cache=CACHE)
def hamerly2(xs: npt.NDArray, ys: npt.NDArray, metric: Metric, maxiter: int):
    n, _ = xs.shape
    k, _ = ys.shape

    # label
    move, upper, lower = _init_labels_hamerly2(xs, ys, metric)
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
        _pdist_hamerly2(ys, metric, pdist, s)
        _label_hamerly2(xs, ys, metric, move, upper, lower, s)

        if not _update_ys(xs, _ys_cp, _ns_cp, move, labels):
            return ys, labels, True

        dys = (_ys_cp / _ns_cp.reshape(-1, 1)) - ys
        _update_bounds(dys, move, upper, lower, metric)

        _ys[:] = _ys_cp
        _ns[:] = _ns_cp
        ys += dys

    return ys, labels, False

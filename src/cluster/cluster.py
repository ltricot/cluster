from numba import njit
import numpy.typing as npt
import numpy as np

from functools import partial

from ._kmeans import (
    _kmeanspp,
    _kmeanspp_approx,
    _kmeanspar,
    _lloyd_exact,
    _lloyd_rtol,
    _lloyd_minibatch,
    _label,
    _label_parallel,
    Metric,
)
from .divergences import l2


class ConvergenceError(RuntimeError):
    ...


_BATCH_THRESHOLD = 30_000
_APPROX_KMEANSPP_THRESHOLD = 10_000
_SAMPLE = 300


@njit
def cost(
    xs: npt.NDArray, ys: npt.NDArray, labels: npt.NDArray[np.int64], metric: Metric
):
    n, _ = xs.shape
    c = 0.0

    for i in range(n):
        c += metric(xs[i], ys[labels[i]]) / n

    return c


class _Cluster:
    @staticmethod
    def __call__(
        xs: npt.NDArray,
        k: int,
        *,
        metric: Metric = l2,
        maxiter: int = 100,
        rtol: float | None = None,
        parallel=True,
    ) -> tuple[npt.NDArray, npt.NDArray[np.int64]]:
        kws = dict(metric=metric, maxiter=maxiter, parallel=parallel)

        if len(xs) > _BATCH_THRESHOLD:
            return _Cluster.batch(xs, k, batch=_BATCH_THRESHOLD, rtol=rtol, **kws)

        return _Cluster.exact(xs, k, **kws)

    @staticmethod
    def _init(xs: npt.NDArray, k: int, /, metric: Metric):
        init = _kmeanspp
        if len(xs) > _APPROX_KMEANSPP_THRESHOLD:
            init = partial(_kmeanspp_approx, sample=k * _SAMPLE)

        return init(xs, k, metric)

    @staticmethod
    def exact(
        xs: npt.NDArray,
        k: int,
        *,
        metric: Metric = l2,
        maxiter: int = 100,
        parallel=True,
    ):
        ys = _Cluster._init(xs, k, metric)

        ys, labels, converged = _lloyd_exact(
            xs, ys, metric, maxiter=maxiter, parallel=parallel
        )

        if not converged:
            raise ConvergenceError

        return ys, labels

    @staticmethod
    def batch(
        xs: npt.NDArray,
        k: int,
        *,
        batch: int,
        metric: Metric = l2,
        maxiter: int = 100,
        rtol: float | None = None,
        parallel=True,
    ):
        ys = _Cluster._init(xs, k, metric)

        ix = np.random.randint(len(xs), size=batch)
        _xs = xs[ix]

        if rtol is None:
            ys, _, converged = _lloyd_exact(
                _xs, ys, metric, maxiter=maxiter, parallel=parallel
            )
        else:
            _rtol = np.float64(rtol)
            ys, _, converged = _lloyd_rtol(
                _xs, ys, metric, maxiter=maxiter, rtol=_rtol, parallel=parallel
            )

        if not converged:
            raise ConvergenceError

        label = _label
        if parallel:
            label = _label_parallel

        labels, _ = label(xs, ys, metric)
        return ys, labels

    @staticmethod
    def minibatch(
        xs: npt.NDArray,
        k: int,
        *,
        batch: int,
        metric: Metric = l2,
        maxiter: int = 100,
        rtol: float | None = None,
        parallel=True,
    ):
        ys = _Cluster._init(xs, k, metric)

        if rtol is None or rtol == 0.0:
            raise ValueError(
                f"rtol value should be > 0 for minibatch K-Means, but is {rtol}"
            )

        _rtol = np.float64(rtol)
        ys, _, converged = _lloyd_minibatch(
            xs, ys, metric, maxiter=maxiter, rtol=_rtol, batch=batch, parallel=parallel
        )

        if not converged:
            raise ConvergenceError

        label = _label
        if parallel:
            label = _label_parallel

        labels, _ = label(xs, ys, metric)
        return ys, labels


cluster = _Cluster()

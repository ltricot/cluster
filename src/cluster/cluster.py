import numpy.typing as npt

from functools import partial

from ._kmeans import _kmeanspp, _kmeanspp_approx, _lloyd1, Metric
from .divergences import l2


class ConvergenceError(RuntimeError):
    ...


_THRESHOLD = 10000
_SAMPLE = 30


def cluster(
    xs: npt.NDArray, k: int, metric: Metric = l2, maxiter: int = 100
) -> npt.NDArray:
    init = _kmeanspp
    if len(xs) > _THRESHOLD:
        init = partial(_kmeanspp_approx, sample=k * _SAMPLE)

    ys = init(xs, k, metric)
    ys, converged = _lloyd1(xs, ys, metric, maxiter)
    if not converged:
        raise ConvergenceError

    return ys

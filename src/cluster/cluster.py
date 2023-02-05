import numpy.typing as npt
import numpy as np

from functools import partial

from ._kmeans import _kmeanspp, _kmeanspp_approx, _lloyd, Metric
from .divergences import l2


class ConvergenceError(RuntimeError):
    ...


_BATCH_THRESHOLD = 30_000
_APPROX_KMEANSPP_THRESHOLD = 10_000
_SAMPLE = 30


def cluster(
    xs: npt.NDArray, k: int, metric: Metric = l2, maxiter: int = 100
) -> npt.NDArray:
    init = _kmeanspp
    if len(xs) > _APPROX_KMEANSPP_THRESHOLD:
        init = partial(_kmeanspp_approx, sample=k * _SAMPLE)

    ys = init(xs, k, metric)

    if len(xs) > _BATCH_THRESHOLD:
        ix = np.random.randint(len(xs), size=_BATCH_THRESHOLD)
        xs = xs[ix]

    ys, converged = _lloyd(xs, ys, metric, maxiter)
    if not converged:
        raise ConvergenceError

    return ys

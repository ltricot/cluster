import numpy.typing as npt
import numpy as np

from .methods.initialization import mckmeanspp
from .methods import hamerly2
from . import metrics


MAXITER = 1000


class ConvergenceError(RuntimeError):
    ...


def cluster(
    xs: npt.NDArray, k: int, metric: metrics.Metric = metrics.l2, maxiter=MAXITER
):
    n, d = xs.shape

    sample = 10 * np.log(n) * np.sqrt(10 + d) * k
    sample = min(sample, 300)
    sample = max(sample, 30)
    ys = mckmeanspp(xs, k, metric, sample=sample)

    ys, labels, converged = hamerly2(xs, ys, metric, maxiter=maxiter)

    if not converged:
        raise ConvergenceError

    return ys, labels

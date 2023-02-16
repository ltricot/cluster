import numpy.typing as npt
import numpy as np

from typing import Protocol
from typing import Optional

from .methods.initialization import mckmeanspp
from .methods import hamerly2, compare
from . import metrics


MAXITER = 1000


class ConvergenceError(RuntimeError):
    ...


class Method(Protocol):
    def __call__(
        self, xs: npt.NDArray, ys: npt.NDArray, metric: metrics.Metric, maxiter: int
    ) -> tuple[npt.NDArray, npt.NDArray, bool]:
        ...


def cluster(
    xs: npt.NDArray,
    k: int,
    metric: metrics.Metric = metrics.l2,
    method: Optional[Method] = None,
    maxiter=MAXITER,
) -> tuple[npt.NDArray, npt.NDArray]:
    n, d = xs.shape

    # hardly a sure thing
    sample = 10 * np.log(n) * np.sqrt(10 + d) * k
    sample = min(sample, 300)
    sample = max(sample, 30)
    ys = mckmeanspp(xs, k, metric, sample=sample)

    # yet again, hardly a sure thing
    if method is None:
        _method = hamerly2
        if n / k <= 128:
            _method = compare
    else:
        _method = method

    ys, labels, converged = _method(xs, ys, metric, maxiter=maxiter)
    if not converged:
        raise ConvergenceError

    return ys, labels

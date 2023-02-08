from numba import njit
import numpy.typing as npt
import numpy as np

from typing import Protocol

from ._settings import CACHE


class Metric(Protocol):
    def __call__(self, x: npt.NDArray, y: npt.NDArray) -> float:
        ...


@njit(cache=CACHE, nogil=True)
def l2(x: npt.NDArray, y: npt.NDArray) -> float:
    return np.sqrt(np.sum((x - y) ** 2))


@njit(cache=CACHE, nogil=True)
def emd(x: npt.NDArray, y: npt.NDArray) -> float:
    cost, carry = 0.0, 0.0

    for i in range(x.shape[0]):
        carry += x[i] - y[i]
        cost += np.abs(carry)

    return cost

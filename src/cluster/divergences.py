import numpy.typing as npt
import numpy as np
from numba import njit  # type: ignore


@njit
def l2(x: npt.NDArray, y: npt.NDArray) -> float:
    return np.linalg.norm(x - y)  # type: ignore


@njit
def emd1(x: npt.NDArray, y: npt.NDArray) -> float:
    carry, cost = 0.0, 0.0

    for i in range(x.shape[0]):
        carry = carry + x[i] - y[i]
        cost += np.abs(carry)

    return cost

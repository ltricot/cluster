import numpy.typing as npt
import numpy as np
from numba import njit  # type: ignore


@njit(nogil=True)
def l2(x: npt.NDArray, y: npt.NDArray) -> np.float64:
    c = np.float64(0)

    for i in range(len(x)):
        c += (x[i] - y[i]) ** 2

    return np.sqrt(c)


@njit(nogil=True)
def emd1(x: npt.NDArray, y: npt.NDArray) -> np.float64:
    carry, cost = 0.0, np.float64(0.0)

    for i in range(x.shape[0]):
        carry = carry + x[i] - y[i]
        cost += np.abs(carry)

    return cost

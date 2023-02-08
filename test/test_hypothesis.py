import hypothesis.extra.numpy as hnp
from hypothesis import strategies as st
from hypothesis import assume, given, settings, HealthCheck
import numpy.typing as npt
import numpy as np

import warnings

from cluster.methods import lloyd, hamerly, compare, sort
from cluster import metrics


methods = [
    lloyd,
    hamerly,
    compare,
    sort,
]


@given(
    xs=hnp.arrays(
        dtype=np.float64,
        shape=hnp.array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=30),
        elements={"allow_nan": False, "allow_infinity": False},
        unique=True,
    ),
    k=st.integers(min_value=1, max_value=8),
)
@settings(deadline=None, suppress_health_check=(HealthCheck.data_too_large,))
def test_methods(xs: npt.NDArray[np.float64], k: int):
    n, d = xs.shape

    # this fun trick makes sure there are no ties
    # otherwise the `sort` method, which inspects centers in
    # a different order than the others, won't have the same results
    with warnings.catch_warnings():
        warnings.filterwarnings("error")

        try:
            xs = xs.cumsum(axis=0)
        except RuntimeWarning:
            assume(False)

    assume(n <= 30)
    assume(d <= 8)
    assume(k <= n)

    ys = xs[:k]

    lbls = []
    for method in methods:
        lbls.append(method(xs, ys, metrics.l2, 1000))

    reference = lbls[0]
    for lbl, method in zip(lbls, methods):
        assert len(lbl) == xs.shape[0], (method.__name__, xs)
        assert np.array_equal(lbl, reference), (method.__name__, xs)

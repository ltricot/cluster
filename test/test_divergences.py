import numpy as np

from cluster.divergences import l2, emd1


def test_l2():
    x, y = np.array([1.0, 2, 3]), np.array([4.0, 5, 6])
    assert np.allclose(l2(x, y), 5.19615242271)


def test_emd():
    x, y = np.array([0.1, 0.1, 0.8]), np.array([0.1, 0.8, 0.1])
    assert np.allclose(emd1(x, y), 0.7)

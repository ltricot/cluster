import numpy as np

from cluster import cluster


def test_cluster():
    xs = np.asarray(
        [
            [1.0, 2, 3],
            [1.1, 2.1, 3.1],
            [4.0, 5, 6],
        ]
    )

    ys = cluster(xs, 2)
    if np.linalg.norm(ys[0]) > np.linalg.norm(ys[1]):
        ys = ys[::-1]

    assert np.allclose(ys, np.asarray([[1.05, 2.05, 3.05], [4.0, 5.0, 6.0]]))

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

    _, labels = cluster(xs, 2, parallel=False)
    l1, l2, l3 = labels

    assert l1 == l2
    assert l1 != l3

# One True Cluster

## Introduction

This package provides a `cluster` function with the following signature:
```python
def cluster(
    xs: npt.NDArray, k: int, metric: metrics.Metric = metrics.l2, maxiter=MAXITER
) -> tuple[npt.NDArray, npt.NDArray]:
```

The returned tuple contains the centroids and the labels.

The `Metric` type is a protocol defined thus:
```python
class Metric(Protocol):
    def __call__(self, x: npt.NDArray, y: npt.NDArray) -> float:
        ...
```

Metrics must be implemented in Numba.

`cluster` uses [Hamerly's algorithm](https://cs.baylor.edu/~hamerly/papers/sdm_2010.pdf) to compute Lloyd's iteration. Initialization is performed using a version of K-Means++ wherein candidate clusters are sampled from a subset of the dataset. Relying on the law of large numbers.

## TODO

- [ ] Generic version of Hamerly & LLoyd which do not use Numba to allow the clustering of arbitrary objects
- [ ] Triangular & non triangular metrics, using a decorator ; different metrics use different algorithms
- [ ] Implement some bregman divergences
- [ ] Choose different algo based on problem instance characteristics

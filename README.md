# Clustering with Bregman Divergences

## Introduction

This package provides a `cluster` function with the following signature:
```python
def cluster(
    xs: npt.NDArray, k: int, metric: Metric | None, maxiter: int = 100
) -> npt.NDArray:
    ...
```

The `Metric` type is a protocol defined thus:
```python
class Metric(Protocol):
    def __call__(self, x: npt.NDArray, y: npt.NDArray) -> float:
        ...
```

Behind the `cluster` method is the K-Means algorithm, as studied in the paper [Clustering with Bregman Divergences](https://www.jmlr.org/papers/volume6/banerjee05b/banerjee05b.pdf). Metrics must be bregman divergences for the K-Means algorithm to be correct.

For small datasets, `cluster` initializes centroids using K-Means++. When the dataset is relatively large, we approximate K-Means++ by sampling from the dataset rather than considering all points as candidates.

If K-Means doesn't convege within `maxiter` iterations, `cluster` will throw a `ConvergenceError`.

## TODO

- [ ] [K-Means|| initialization](https://www.ccs.neu.edu/home/radivojac/classes/2021fallcs6220/hamerly_bookchapter_2014.pdf)
- [ ] Parallel version of Lloyd's iteration ; for some reason simply asking numba to parallelize the outer loop yields a large slowdown
- [ ] [Hamerly's triangle inequality optimization](https://www.ccs.neu.edu/home/radivojac/classes/2021fallcs6220/hamerly_bookchapter_2014.pdf) for Bregman divergences which satisfy the triangle inequality (notably l2)

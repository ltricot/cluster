# One True Cluster

## Introduction

This package provides a `cluster` function with the following signature:
```python
def cluster(
    xs: npt.NDArray,
    k: int,
    metric: metrics.Metric = metrics.l2,
    method: Optional[Method] = None,
    maxiter=MAXITER,
) -> tuple[npt.NDArray, npt.NDArray]:
```

The returned tuple contains the centroids and the labels.

The `Metric` and `Method` types are protocols defined thus:
```python
class Metric(Protocol):
    def __call__(self, x: npt.NDArray, y: npt.NDArray) -> float:
        ...

class Method(Protocol):
    def __call__(
        self, xs: npt.NDArray, ys: npt.NDArray, metric: metrics.Metric, maxiter: int
    ) -> tuple[npt.NDArray, npt.NDArray, bool]:
        ...
```

Metrics must be implemented in Numba. Methods are implemented by the package and there should generally be no need to specify one.

`cluster` uses [Hamerly's algorithm](https://cs.baylor.edu/~hamerly/papers/sdm_2010.pdf) to compute the Lloyd iteration when $n / k$ is large, and [Philips's compare algorithm](https://www.semanticscholar.org/paper/Acceleration-of-K-Means-and-Related-Clustering-Phillips/badb2fb3c8792d5b70aa27ae1ae231208ba4253f) when $n / k$ is small. Initialization is performed using a version of K-Means++ wherein candidate clusters are sampled from a subset of the dataset. Relying on the law of large numbers.

## Some observations

- Hamerly isn't always the fastest algorithm ; it is advantaged as $n / k$ grows but Philips's algorithms (sort and compare) are better as $n / k$ decreases
- Initialization methods developed since K-Means++ are slower, and do not seem more accurate, than accelerating K-Means++ by sampling from a random subset of the dataset ; I do not know whether some datasets can be problematic for this method

## TODO

- [ ] Generic version of Hamerly & LLoyd which do not use Numba to allow the clustering of arbitrary objects
    - Must not only have a generic metric but also a generic `prototype` (which currently is np.mean) ; implementing this would mean different top level routines (lloyd, hamerly, etc...), which don't use the vector sum trick to update centroids and call `prototype` instead
    - This will notably be useful for sparse data !
    - [ ] Sparse data example
    - [ ] Randomized Linear Algebra for very high dimensional data ?
- [ ] Triangular & non triangular metrics, using a decorator ; different metrics use different algorithms
- [ ] Implement some bregman divergences

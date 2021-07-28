# Python-GTSNE

Python library containing GTSNE algorithms.

Global t-Distributed Stochastic Neighbor Embedding

## Installation

## Requirements

- [cblas](http://www.netlib.org/blas/) or [openblas](https://github.com/xianyi/OpenBLAS).
Tested version is v0.2.5 and v0.2.6 (not necessary for OSX).

From PyPI:

```
pip install gtsne
```

## Usage

Basic usage:

```python
from gtsne import gtsne
X_2d = gtsne(X)
```

### Examples

- [Iris](http://nbviewer.ipython.org/urls/raw.github.com/danielfrg/py_tsne/master/examples/iris.ipynb)

## Algorithms

## Acknowledgements
This code is adapted from code [Barnes-Hut-SNE](https://github.com/danielfrg/tsne)
Special thanks to Laurens van der Maaten and Daniel Rodriguez. 

## Additional resources

- See *Barnes-Hut-SNE* (2013), L.J.P. van der Maaten. It is available on [arxiv](http://arxiv.org/abs/1301.3342).

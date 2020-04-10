import numpy as np
from Error import ShapeIncompatible


def mean_square_error(x: np.ndarray, y: np.ndarray):
  if x.shape != y.shape:
    raise ShapeIncompatible("x and y's sizes are different")
  return np.mean(np.square(x - y))


def cross_entropy(x, y):
  if x.shape != y.shape:
    raise ShapeIncompatible("Size Different For (x: {}, y: {})".format(x.shape, y.shape))
  return np.mean(np.sum(y * np.log(x), axis=1))




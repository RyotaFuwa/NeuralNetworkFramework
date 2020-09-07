from abc import ABC
import numpy as np
from _layers import SequentialLayer


class Activation(SequentialLayer, ABC):
  LEARNABLE = False
  ONLY_IN_TRAINING = False

  def __init__(self, i: SequentialLayer = None):
    super().__init__()
    if i is not None:
      self.__call__(i)
    else:
      self._shape = ()

  def __call__(self, i):
    super().__call__(i)
    self._shape = i.shape


class Sigmoid(Activation):
  y: np.ndarray

  def f(self, x: np.ndarray) -> np.ndarray:
    self.y = 1.0 / (1.0 + np.exp(-x))
    return self.y

  def df(self, dy: np.ndarray) -> np.ndarray:
    return self.y * (1.0 - self.y) * dy


class ReLU(Activation):
  x: np.ndarray

  def f(self, x: np.ndarray) -> np.ndarray:
    self.x = x
    return np.where(x > 0.0, x, 0.0)

  def df(self, dy: np.ndarray) -> np.ndarray:
    k = np.where(self.x > 0.0, 1.0, 0.0) * dy
    return k


class Tanh(Activation):
  x: np.ndarray

  def f(self, x: np.ndarray) -> np.ndarray:
    self.x = x
    return np.tanh(x)

  def df(self, dy: np.ndarray) -> np.ndarray:
    return 1.0 / np.square(np.cosh(self.x)) * dy


class Softmax(Activation):  # Assuming the dim of input is 2 (sample, dim_of_data)
  y: np.ndarray

  def f(self, x: np.ndarray) -> np.ndarray:
    max_v = np.max(x, axis=-1).reshape((*x.shape[:-1], 1))
    self.y = np.exp(x - max_v) / np.sum(np.exp(x - max_v), axis=-1).reshape((*x.shape[:-1], 1))
    return self.y

  def df(self, dy: np.ndarray) -> np.ndarray:
    num_samples = self.y.shape[0]
    num_classes = self.y.shape[-1]
    out = []
    for i in range(num_samples):  # iterate over each sample in min-batch
      y = self.y[i].reshape((1, -1))
      M = np.dot(y.T, y)
      M -= np.diag(M)
      for j in range(num_classes):  # set diagonal element to yi(1 - yi)
        yi = self.y[i, j]
        M[j, j] = yi * (1.0 - yi)
      out.append(np.dot(dy[i], M))
    return np.array(out)


class Linear(Activation):
  def f(self, x: np.ndarray) -> np.ndarray:
    return x

  def df(self, dy: np.ndarray) -> np.ndarray:
    return dy

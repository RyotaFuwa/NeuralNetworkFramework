from abc import ABC
import numpy as np
from _layer import _Layer


class Activation(_Layer, ABC):
  def __init__(self):
    super().__init__()

  def __call__(self, i):
    self.shape = i.shape
    super().__call__(i)


class Linear(Activation):
  def f(self, x: np.ndarray, training: bool) -> np.ndarray:
    return x

  def df(self, dy: np.ndarray) -> np.ndarray:
    return dy


class Sigmoid(Activation):
  y: np.ndarray

  def f(self, x: np.ndarray, training: bool) -> np.ndarray:
    self.y = 1.0 / (1.0 + np.exp(-x))
    return self.y

  def df(self, dy: np.ndarray) -> np.ndarray:
    return self.y * (1.0 - self.y) * dy


class ReLU(Activation):
  mask: np.ndarray

  def f(self, x: np.ndarray, training: bool) -> np.ndarray:
    self.mask = x <= 0
    out = x.copy()
    out[self.mask] = 0
    return out

  def df(self, dy: np.ndarray) -> np.ndarray:
    dy[self.mask] = 0
    return dy


class Tanh(Activation):
  x: np.ndarray

  def f(self, x: np.ndarray, training: bool) -> np.ndarray:
    self.x = x
    return np.tanh(x)

  def df(self, dy: np.ndarray) -> np.ndarray:
    return 1.0 / np.square(np.cosh(self.x)) * dy


class Softmax(Activation):
  y: np.ndarray

  def f(self, x: np.ndarray, training: bool) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    self.y = np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
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


REGISTERED_ACTIVATION = {
  'linear': Linear,
  'sigmoid': Sigmoid,
  'relu': ReLU,
  'tanh': Tanh,
  'softmax': Softmax,
}


def activation_loader(key: str = ''):
  if key in REGISTERED_ACTIVATION:
    return REGISTERED_ACTIVATION[key]()
  else:
    return None

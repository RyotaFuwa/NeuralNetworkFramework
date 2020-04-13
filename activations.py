from _layers import SequenceLayer
from abc import ABC
import numpy as np


class Activation(SequenceLayer, ABC):
  LEARNABLE = False
  ONLY_IN_TRAINING = False
  x: np.ndarray
  y: np.ndarray
  def __init__(self, i: SequenceLayer):
    if i is not None:
      self.__call__(i)

  def __call__(self, i):
    super().__call__(i)


class Sigmoid(Activation):
  def f(self, x: np.ndarray) -> np.ndarray:
    self.x = x
    self.y = 1 / (1 + np.exp(-x))
    return self.y

  def df(self, dy: np.ndarray) -> np.ndarray:
    return self.y * (1 - self.y) * dy


class ReLU(Activation):
  def f(self, x: np.ndarray) -> np.ndarray:
    self.x = x
    self.y = np.where(x > 0, x, 0)
    return self.y

  def df(self, dy: np.ndarray) -> np.ndarray:
    return np.where(self.x > 0, 1, 0) * dy


class Tanh(Activation):
  def f(self, x: np.ndarray) -> np.ndarray:
    self.x = x
    self.y = np.tanh(x)
    return self.y

  def df(self, dy: np.ndarray) -> np.ndarray:
    return 1 / np.square(np.cosh(self.x)) * dy


class Softmax(Activation):  # TODO
  def f(self, x: np.ndarray) -> np.ndarray:
    self.x = x
    max_v = np.max(x)
    self.y = np.exp(x - max_v) / np.sum(np.exp(x - max_v))
    return self.y

  def df(self, dy: np.ndarray) -> np.ndarray:
    D = np.zeros((self.y.shape[0], self.y.shape[1], self.y.shape[1]))
    for i in range(self.y.shape[0]):
      D[i] = -self.y[i].dot(self.y[i])
      D[i] -= np.diag(D[i])
      for j in range(D.shape[0]):
        yi = self.y[i, j]
        D[i, j, j] = yi * (1 - yi)
      return dy[i].dot(D[i])


class Linear(Activation):
  def f(self, x: np.ndarray) -> np.ndarray:
    self.x = x
    self.y = x
    return self.y
  def df(self, dy: np.ndarray) -> np.ndarray:
    return dy


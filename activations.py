from layers import Layer
from abc import ABC
import numpy as np


class Activation(Layer, ABC):
  LEARNABLE = False
  ONLY_IN_TRAINING = False


class Sigmoid(Activation):
  def f(self, x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

  def df(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return y * (1 - y)


class ReLU(Activation):
  def f(self, x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, 0)

  def df(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0)


class Tanh(Activation):
  def f(self, x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

  def df(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 1 / np.square(np.cosh(x))


class Softmax(Activation):
  def f(self, x: np.ndarray) -> np.ndarray:
    max_v = np.max(x)
    return np.exp(x - max_v) / np.sum(np.exp(x - max_v))
  def df(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.ones_like(x)


class Linear(Activation):
  def f(self, x: np.ndarray) -> np.ndarray:
    return x
  def df(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.ones_like(y)


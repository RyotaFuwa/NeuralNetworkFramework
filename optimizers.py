from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from _layers import Layer
from queue import Queue
from misc import W

EPSILON = np.finfo(float).eps


"""Optimizer Design
an instance of this class can store the reference to the weights of all layers in the model
as well as other states such as dW depending on the type of optimizer
"""


class Optimizer(ABC):
  _layers: Tuple[Layer]

  @abstractmethod
  def __init__(self, *args, **kwargs):
    """set parameters"""

  @abstractmethod
  def update(self, in_batch, cost):
    """update parameters"""

  def initialize(self, layers):
    self._layers = layers

  def W(self):
    return np.array([l.w for l in self._layers])

  def dW(self):
    return np.array([l.dw for l in self._layers])

  @staticmethod
  def random_like(w: np.ndarray, mue=0.0, sigma=1.0):
    def helper(w: np.ndarray, place: list):
      if w.dtype == 'O':
        new_place = []
        for obj in w:
          helper(obj, new_place)
        place.append(np.array(new_place))
      else:
        value = (np.random.randn(*w.shape) - mue) * sigma
        place.append(value)
    out = []
    helper(w, out)
    return np.array(out[0])


class RandomOptimizer(Optimizer):
  w: np.ndarray

  def __init__(self, learning_rate):
    self.learning_rate = learning_rate

  def initialize(self, layers):
    super().initialize(layers)
    self.w = self.W()

  def update(self):
      self.w += self.learning_rate * Optimizer.random_like(self.w)


class SGD(Optimizer):
  w: np.ndarray
  dw: np.ndarray
  momentum: float

  def __init__(self, learning_rate, momentum=-1):
    self.learning_rate = learning_rate
    self.momentum = momentum

  def initialize(self, layers):
    super().initialize(layers)
    self.w = self.W()
    self.dw = self.dW()

  def update(self):
    if self.momentum > 0:
      self.dw = self.momentum * self.dw - self.learning_rate * self.dW()
      self.w += self.dw
    else:
      self.dw = self.dW()
      self.w -= self.learning_rate * self.dw


class Adam(Optimizer):  # TODO
  def __init__(self, learning_rate, beta1, beta2):
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.time = 0

  def initialize(self, layers):
    super().initialize(layers)
    self.w = self.W()
    self.dw = self.dW()
    self.m = Optimizer.random_like(self.w, mue=0, sigma=0.001)
    self.v = Optimizer.random_like(self.w, mue=0, sigma=0.001)

  def update(self):
    self.time += 1
    self.dw = self.dW()
    self.m = self.beta1 * self.m + (1.0 - self.beta1) * self.dw
    self.v = self.beta2 * self.v + (1.0 - self.beta2) * self.dw ** 2

    m_hat = self.m / (1.0 - self.beta1 ** self.time)
    v_hat = self.v / (1.0 - self.beta2 ** self.time)
    self.w -= self.learning_rate * m_hat / (v_hat + EPSILON)



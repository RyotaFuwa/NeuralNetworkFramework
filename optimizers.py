from abc import ABC, abstractmethod
import numpy as np
from misc import random_like

EPSILON = np.finfo(float).eps

"""Optimizer Design
an instance of this class can store the reference to the weights of all layers in the model
as well as other states such as dW depending on the type of optimizer
"""


class Updater(ABC):
  @abstractmethod
  def update(self, w, dw):
    """update w"""

  def __call__(self, w, dw):
    self.update(w, dw)


class RandomUpdater(Updater):
  def __init__(self, w, learning_rate):
    self.learning_rate = learning_rate

  def update(self, w, dw):
    w -= self.learning_rate * random_like(self.w)


class SGDUpdater(Updater):
  def __init__(self, layer, learning_rate, momentum=-1.0):
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.V = np.random.randn(*layer.w.shape) * 0.01

  def update(self, w, dw):
    self.V = self.momentum * self.V - self.learning_rate * dw
    w += self.V


class AdamUpdater(Updater):
  def __init__(self, layer, learning_rate, beta1, beta2):
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.m = np.random.randn(*layer.w.shape) * 0.01
    self.v = np.abs(np.random.randn(*layer.w.shape)) * 0.01
    self.time = 0

  def update(self, w: np.ndarray, dw: np.ndarray):
    self.time += 1
    self.m = self.beta1 * self.m + (1.0 - self.beta1) * dw
    self.v = self.beta2 * self.v + (1.0 - self.beta2) * np.square(dw)

    m_hat = self.m / (1.0 - np.power(self.beta1, self.time))
    v_hat = self.v / (1.0 - np.power(self.beta2, self.time))
    k = self.learning_rate * m_hat / (np.sqrt(v_hat) + EPSILON)
    w -= k


class Optimizer(ABC):
  @abstractmethod
  def __init__(self, *args, **kwargs):
    """set parameters"""

  @abstractmethod
  def __call__(self, layer):
    """return instance of Updater class defined in Optimizer class"""


class RandomOptimizer(Optimizer):
  learning_rate: float

  def __init__(self, learning_rate, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.learning_rate = learning_rate

  def __call__(self, layer):
    return RandomUpdater(layer.w, self.learning_rate)


class SGD(Optimizer):
  V: np.ndarray
  momentum: float

  def __init__(self, learning_rate=0.01, momentum=0.0, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.learning_rate = learning_rate
    self.momentum = momentum

  def __call__(self, w):
    return SGDUpdater(w, self.learning_rate, self.momentum)


class Adam(Optimizer):
  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.time = 0

  def __call__(self, w):
    return AdamUpdater(w, self.learning_rate, self.beta1, self.beta2)



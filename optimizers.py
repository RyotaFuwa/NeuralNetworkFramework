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
  def __init__(self, w, learning_rate, momentum=-1.0):
    self.learning_rate = learning_rate
    self.momentum = momentum
    if momentum > 0:
      self.V = np.random.randn(*w.shape) * 0.0001

  def update(self, w, dw):
    if self.momentum > 0:
      self.V = self.momentum * self.V - self.learning_rate * dw
      w += self.V
    else:
      w -= self.learning_rate * dw


class AdamUpdater(Updater):
  def __init__(self, w, learning_rate, beta1, beta2):
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.m = random_like(w, mue=0, sigma=0.001)
    self.v = random_like(w, mue=0, sigma=0.001)
    self.time = 0

  def update(self, w, dw):
    self.time += 1
    self.m = self.beta1 * self.m + (1.0 - self.beta1) * dw
    self.v = self.beta2 * self.v + (1.0 - self.beta2) * dw ** 2

    m_hat = self.m / (1.0 - self.beta1 ** self.time)
    v_hat = self.v / (1.0 - self.beta2 ** self.time)
    w -= self.learning_rate * m_hat / (v_hat ** 0.5 + EPSILON)


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

  def __init__(self, learning_rate, momentum=-1, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.learning_rate = learning_rate
    self.momentum = momentum

  def __call__(self, w):
    return SGDUpdater(w, self.learning_rate, self.momentum)


class Adam(Optimizer):
  def __init__(self, learning_rate, beta1, beta2, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.time = 0

  def __call__(self, w):
    return AdamUpdater(w, self.learning_rate, self.beta1, self.beta2)



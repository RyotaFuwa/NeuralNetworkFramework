from abc import ABC, abstractmethod
import numpy as np
from misc import random_like

EPSILON = np.finfo(float).eps

"""Optimizer Design
an instance of this class can store the reference to the weights of all layers in the model
as well as other states such as dW depending on the type of optimizer
"""


class Updater(ABC):
  def __init__(self, *args, **kwargs):
    self.clipnorm = kwargs.get('clipnorm')
    self.clipvalue = kwargs.get('clipvalue')
    self.epsilon = kwargs.get('epsilon', 1e-07)

  @abstractmethod
  def update(self, w, dw):
    """update w based on w, dw, and state"""

  def clip(self, dw):
    if self.clipnorm is not None:
      ratio = np.linalg.norm(dw, axis=1).reshape((-1, 1))
      ratio = np.where(ratio > self.clipnorm, self.clipnorm / (ratio + self.epsilon), 1.0)
      dw *= ratio
    elif self.clipvalue is not None:
      dw = np.clip(dw, -self.clipvalue, self.clipvalue)
    return dw

  def __call__(self, w, dw):
    self.update(w, dw)


class RandomUpdater(Updater):
  def __init__(self, w, learning_rate, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.learning_rate = learning_rate

  def update(self, w, dw):
    w -= self.learning_rate * random_like(self.w)


class SGDUpdater(Updater):
  def __init__(self, layer, learning_rate, momentum, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.V = np.zeros_like(layer.w)

  def update(self, w, dw):
    dw = self.clip(dw)
    self.V = self.momentum * self.V - self.learning_rate * dw
    w += self.V


class AdamUpdater(Updater):
  def __init__(self, layer, learning_rate, beta1, beta2, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.learning_rate = learning_rate

    self.beta1 = beta1
    self.beta2 = beta2

    self.m = np.zeros_like(layer.w)
    self.v = np.zeros_like(layer.w)
    self.time = 0

  def update(self, w: np.ndarray, dw: np.ndarray):
    dw = self.clip(dw)

    self.time += 1
    self.m = self.beta1 * self.m + (1.0 - self.beta1) * dw
    self.v = self.beta2 * self.v + (1.0 - self.beta2) * np.square(dw)

    beta1_t = np.power(self.beta1, self.time)
    beta2_t = np.power(self.beta2, self.time)

    m_hat = self.m / (1.0 - beta1_t)
    v_hat = self.v / (1.0 - beta2_t)

    w -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class Optimizer(ABC):
  @abstractmethod
  def __init__(self, *args, **kwargs):
    self.args = args
    self.kwargs = kwargs

  @abstractmethod
  def __call__(self, layer):
    """return instance of Updater class defined in Optimizer class"""


class RandomOptimizer(Optimizer):
  learning_rate: float

  def __init__(self, learning_rate, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.learning_rate = learning_rate

  def __call__(self, layer):
    return RandomUpdater(layer.w, self.learning_rate, *self.args, **self.kwargs)


class SGD(Optimizer):
  V: np.ndarray
  momentum: float

  def __init__(self, learning_rate=0.01, momentum=0.0, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.learning_rate = learning_rate
    self.momentum = momentum

  def __call__(self, w):
    return SGDUpdater(w, self.learning_rate, self.momentum, *self.args, **self.kwargs)


class Adam(Optimizer):
  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.time = 0

  def __call__(self, w):
    return AdamUpdater(w, self.learning_rate, self.beta1, self.beta2, *self.args, **self.kwargs)



from abc import ABC, abstractmethod
import numpy as np

from layers import Layer
from misc.constants import EPSILON


class Updater(ABC):

  def __init__(self, optimizer, layer):
    self.optimizer = optimizer
    self.initialize_weights(layer)

  @abstractmethod
  def initialize_weights(self, layer: Layer):
    """initialize weights for the layer"""

  @abstractmethod
  def update(self, w, dw):
    """update w based on w, dw, and state"""

  def clip(self, dw):
    clipnorm = self.optimizer.clipnorm
    clipvalue = self.optimizer.clipvalue

    if clipnorm is not None:
      ratio = np.linalg.norm(dw, axis=1).reshape((-1, 1))
      ratio = np.where(ratio > clipnorm, clipnorm / (ratio + EPSILON), 1.0)
      dw *= ratio
    elif clipvalue is not None:
      dw = np.clip(dw, -clipvalue, clipvalue)
    return dw

  def __call__(self, w, dw):
    self.update(w, dw)


class SGDUpdater(Updater):
  V: np.ndarray

  def __init__(self, optimizer, layer):
    super().__init__(optimizer, layer)

  def initialize_weights(self, layer: Layer):
    self.V = np.zeros_like(layer.w)

  def update(self, w, dw):
    momentum = self.optimizer.momentum
    learning_rate = self.optimizer.learning_rate

    dw = self.clip(dw)
    self.V = momentum * self.V - learning_rate * dw
    w += self.V


class AdamUpdater(Updater):
  m: np.ndarray
  v: np.ndarray

  def __init__(self, optimizer, layer):
    super().__init__(optimizer, layer)

  def initialize_weights(self, layer):
    self.m = np.zeros_like(layer.w)
    self.v = np.zeros_like(layer.w)

  def update(self, w: np.ndarray, dw: np.ndarray):
    dw = self.clip(dw)

    beta1 = self.optimizer.beta1
    beta2 = self.optimizer.beta2
    step = self.optimizer.step
    learning_rate = self.optimizer.learning_rate

    self.m += (1.0 - beta1) * (dw - self.m)
    self.v += (1.0 - beta2) * (np.square(dw) - self.v)

    coefficient = learning_rate * np.sqrt(1.0 - beta2 ** step) / (1.0 - beta1 ** step)

    k = coefficient * self.m
    l = (np.sqrt(self.v) + EPSILON)
    w -= k / l


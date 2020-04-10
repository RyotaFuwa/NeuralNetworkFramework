from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from layers import Layer
from misc import TYPE_LAYER, TYPE_INPUT
from queue import Queue


"""Optimizer Design
an instance of this class can store the reference to the weights of all layers in the model
as well as other states such as dW depending on the type of optimizer
"""


class Optimizer(ABC):

  @abstractmethod
  def __init__(self, *args, **kwargs):
    """set parameters"""

  @abstractmethod
  def initialize(self, L):
    """set up necessary parameters"""

  @abstractmethod
  def update(self, in_batch, cost):
    """update parameters"""

  def __call__(self, cost):
    self._update(cost)


class RandomOptimizer(Optimizer):
  def __init__(self, learning_rate):
    self.learning_rate = learning_rate

  def initialize(self, L):
    self.W = (l.W for l in L)

  def update(self, in_batch, cost):
    for w in self.W:
      self.w += self.learning_rate * np.random.rand(*w.shape)


class SGD(Optimizer):
  def __init__(self, learning_rate):
    self.learning_rate = learning_rate

  def update(self, in_batch, cost):
    pass

  def initialize(self, i, o):
    self._set_W(i, o)
    # allocate storage for weights and dw


class Adam(Optimizer):
  def __init__(self, W):
    self.W = W

  def _update(self, in_batch, cost):
    pass



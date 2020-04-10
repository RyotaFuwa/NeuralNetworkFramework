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
  W: np.ndarray  # reference to the weights in the model
  dW: np.ndarray

  @abstractmethod
  def __init__(self, *args, **kwargs):
    """set parameters"""

  @abstractmethod
  def initialize(self, i, o):
    """set up necessary parameters"""

  @abstractmethod
  def update(self, in_batch, cost):
    """update parameters"""

  def __call__(self, cost):
    self._update(cost)

  def _set_W(self, i, o):
    self.W = []
    self.dW = []
    queue = Queue()
    for l in o:
      queue.put(l)
    while queue.not_empty:
      current_layer = queue.get()
      self.w.append(current_layer.W)
      self.dw.append(current_layer.dW)
      if current_layer != (None,):
        queue.put(current_layer.previous)


class RandomOptimizer(Optimizer):
  def __init__(self, learning_rate):
    self.learning_rate = learning_rate

  def initialize(self, i, o):
    self._set_W(i, o)

  def update(self, in_batch, cost):
    for w, dw in zip(self.W, self.dW):
      dw = np.random.rand(*dw.shape)
      w += self.learning_rate * dw


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



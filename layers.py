import numpy as np
from _layers import *
from activations import Activation
from misc import debug


# Forward Layer. Take 1-dim input and return 1-dim output
class Forward(SequenceLayer):
  _A: np.ndarray
  _b: np.ndarray
  activation: Activation

  def __init__(self, w_size: int, activation: Activation = None):
    super().__init__()
    self._shape = (w_size,)
    self._next = activation(self) if activation is not None else None

  def __call__(self, i: Layer):
    if len(i.shape) != 1:  # check if the input shape is compatible
      raise ShapeIncompatible("Forward Layer accepts only 1-dim data as input")
    self._A = np.random.randn(*i.shape, *self._shape) * 0.001  # initialization of weights
    self._b = np.random.randn(1, *self._shape) * 0.001  # initialization of weights
    self._w = np.array([self._A, self._b])
    self._dw = np.zeros_like(self._w)

    self._prev = i
    i._next = self
    return self._next if self._next is not None else self

  def f(self, x: np.ndarray):
    self._x = x
    return x.dot(self._A) + self._b

  def df(self, dy: np.ndarray):
    db = dy
    da = self._x.T.dot(dy)
    self._dw = np.array([da, db])  # dy/dw
    return dy.dot(da.T)  # dy/dx


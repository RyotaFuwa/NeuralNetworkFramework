import numpy as np
from _layers import *
from activations import Activation


# Forward Layer. Take 1-dim input and return 1-dim output
class Forward(SequenceLayer):

  _A: np.ndarray
  _b: np.ndarray
  activation: Activation

  def __init__(self, w_size: int, activation: Activation):
    self._shape = (w_size,)
    self.activation = activation(self)
    super().__init__()

  def __call__(self, i: Layer):
    if len(i.shape) != 1:  # check if the input shape is compatible
      raise ShapeIncompatible("Forward Layer accepts only 1-dim data as input")
    self._A = np.random.randn(*i.shape, *self._shape) * 0.001  # initialization of weights
    self._b = np.random.randn(1, *self.shape) * 0.001  # initialization of weights
    self._w = np.array([self._A, self._b])
    self._dw = np.zeros_like(self._w)
    super().__call__(i)
    return self

  def f(self, x: np.ndarray):
    self._x = np.array([x])
    return self.activation.f(x.dot(self._A) + self._b)

  def df(self, dy: np.ndarray):
    dy_activation = self.activation.df(dy)
    db = dy_activation
    da = self._x[0].T.dot(dy_activation)
    self._dw = np.array([da, db])  # dy/dw
    return dy_activation.dot(da.T)  # dy/dx


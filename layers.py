from Error import ShapeIncompatible
from _layers import *
from activations import Activation
from misc import random_like


class Mat(SequenceLayer):
  def __init__(self, w_size: int):
    self._shape = (w_size,)
    super().__init__()

  def __call__(self, i: Layer):
    if len(i._shape) != 1:  # check if the input shape is compatible
      raise ShapeIncompatible("Forward Layer accepts only 1-dim data as input")
    self._w = np.random.randn(*i._shape, *self._shape) * 0.001  # initialization of weights
    self._dw = random_like(self._w)
    SequenceLayer.__call__(self, i)
    return self

  def f(self, x: np.ndarray):
    self._x = x
    return x.dot(self._w)

  def df(self, dy: np.ndarray):
    self._dw = self._x.T.dot(dy)  # dy/dw
    return self._w.T.dot(dy)  # dy/dx


# Input Layer. Define the input of the model
class Input(SequenceLayer):
  ONLY_IN_TRAINING = True
  LEARNABLE = False

  def __init__(self, shape: TYPE_SHAPE):
    self._prev = None
    if type(shape) == int:
      self._shape = (shape,)
    else:
      self._shape = shape
    super().__init__()

  def __call__(self, i):
    self._next = (i,)

  def f(self, x: np.ndarray):
    return x

  def df(self, dy: np.ndarray):
    pass


# Dropout Layer.
class Dropout(SequenceLayer):
  ONLY_IN_TRAINING = True
  LEARNABLE = False
  _filter: np.ndarray

  def __init__(self, prob):
    self.prob = prob
    super().__init__()

  def __call__(self, i: Layer):
    self._shape = i.shape
    self._filter = np.ones((1, *i.shape))
    super().__call__(i)
    return self

  def f(self, x: np.ndarray):
    self._filter = np.where(np.random.rand(*self._filter.shape) > self.prob, 1.0, 0.0)
    return x * self._filter

  def df(self, dy: np.ndarray):
    return dy * self._filter


# Forward Layer. Take 1-dim input and return 1-dim output
class Forward(SequenceLayer):
  ONLY_IN_TRAINING = False
  LEARNABLE = True
  activation: Activation

  def __init__(self, w_size: int, activation: Activation = None):
    super().__init__()
    self._shape = (w_size,)
    self._next = activation(self) if activation is not None else None

  def __call__(self, i: Layer):
    if len(i.shape) != 1:  # check if the input shape is compatible
      raise ShapeIncompatible("Forward Layer accepts only 1-dim data as input")
    A = np.random.randn(*i.shape, *self._shape) * 0.001  # initialization of weights
    b = np.random.randn(1, *self._shape) * 0.001  # initialization of weights
    self._w = np.concatenate((A, b))
    self._dw = np.random.randn(*self._w.shape) * 0.001

    self._prev = i
    i._next = self
    return self._next if self._next is not None else self

  def f(self, x: np.ndarray):
    self._x = x
    return x.dot(self.w[:-1]) + self.w[-1]

  def df(self, dy: np.ndarray):
    db = dy.sum(axis=0).reshape((1, -1))
    da = self._x.T.dot(dy)
    self._dw = np.concatenate((da, db))  # dy/dw
    self.updater(self.w, self.dw)
    return dy.dot(da.T)  # dy/dx


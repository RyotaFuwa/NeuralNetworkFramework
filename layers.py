from Error import ShapeIncompatible
from _layers import *
from activations import Activation
from misc import random_like


class Mat(LearnableSequential):
  x: np.ndarray

  def __init__(self, w_size: int):
    super().__init__()
    self._shape = (w_size,)

  def __call__(self, i: Layer):
    if len(i._shape) != 1:  # check if the input shape is compatible
      raise ShapeIncompatible("Forward Layer accepts only 1-dim data as input")
    self._w = np.random.randn(*i._shape, *self._shape) * 0.01  # initialization of weights
    self._dw = random_like(self._w)
    return super().__call__(i)

  def f(self, x: np.ndarray):
    self.x = x
    return x.dot(self._w)

  def df(self, dy: np.ndarray):
    self._dw = self.x.T.dot(dy)  # dy/dw
    return self._w.T.dot(dy)  # dy/dx


# Input Layer. Define the input of the model
class Input(SequentialLayer):
  ONLY_IN_TRAINING = True
  LEARNABLE = False

  def __init__(self, shape: TYPE_SHAPE):
    if type(shape) == int:
      self._shape = (shape,)
    else:
      self._shape = shape
    super().__init__()

  def __call__(self, i):
    pass

  def f(self, x: np.ndarray):
    return x

  def df(self, dy: np.ndarray):
    pass


# Dropout Layer.
class Dropout(SequentialLayer):
  ONLY_IN_TRAINING = True
  LEARNABLE = False
  _filter: np.ndarray

  def __init__(self, prob):
    self.prob = prob
    super().__init__()

  def __call__(self, i: Layer):
    self._shape = i.shape
    self._filter = np.ones((1, *i.shape))
    return super().__call__(i)

  def f(self, x: np.ndarray):
    self._filter = np.where(np.random.rand(*self._filter.shape) > self.prob, 1.0, 0.0)
    return x * self._filter

  def df(self, dy: np.ndarray):
    return dy * self._filter


# Forward Layer. Take 1-dim input and return 1-dim output
class Forward(LearnableSequential):
  x: np.ndarray

  def __init__(self, w_size: int, activation: Activation = None):
    self._shape = (w_size,)
    super().__init__(activation)

  def __call__(self, i: Layer):
    if len(i.shape) != 1:  # check if the input shape is compatible
      raise ShapeIncompatible("Forward Layer accepts only 1-dim data as input")
    A = np.random.randn(*i.shape, *self._shape) * 0.01  # initialization of weights
    b = np.random.randn(1, *self._shape) * 0.01  # initialization of weights
    self._w = np.concatenate((A, b))
    self._dw = np.random.randn(*self._w.shape) * 0.01
    return super().__call__(i)

  def f(self, x: np.ndarray):
    self.x = x
    return x.dot(self.w[:-1]) + self.w[-1]

  def df(self, dy: np.ndarray):
    db = dy.sum(axis=0).reshape((1, -1))
    da = self.x.T.dot(dy)
    self._dw = np.concatenate((da, db))  # dy/dw
    self.updater(self.w, self.dw)
    return dy.dot(da.T)  # dy/dx


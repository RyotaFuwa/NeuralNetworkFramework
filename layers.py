from abc import ABC, abstractmethod
import numpy as np
from _layer import _Layer
from misc.errors import ShapeIncompatible, NetworkNotReady
from activations import activation_loader, Activation
from misc.types import SHAPE, shape2tuple
from optimizers import Updater


class Layer(_Layer, ABC):
  """layer with learnable weights."""
  _w: np.ndarray  # w: learnable weights
  _dw: np.ndarray  # dw: dL/dw
  updater: Updater

  @property
  def w(self):
    pass

  @w.getter
  def w(self):
    return self._w

  @property
  def dw(self):
    pass

  @dw.getter
  def dw(self):
    return self._dw

  @abstractmethod
  def __init__(self, activation: str = ''):
    super().__init__()
    self.next = activation_loader(activation)

  @abstractmethod
  def __call__(self, i: _Layer) -> _Layer:
    if isinstance(self.next, Activation):
      self.next(self)
    return super().__call__(i)

  @abstractmethod
  def f(self, x: np.ndarray, training: bool) -> np.ndarray:
    """forward calculation"""

  @abstractmethod
  def df(self, x: np.ndarray) -> np.ndarray:
    """back propagation"""

  @abstractmethod
  def initialize(self):
    """initialize weights"""


class Input(_Layer):
  def __init__(self, shape: SHAPE):
    super().__init__()
    self.shape = shape2tuple(shape)

  def __call__(self, i: _Layer):
    raise Exception('Input Layer is not supposed to be in the middle of a network')

  def f(self, x: np.ndarray, training: bool) -> np.ndarray:
    return x

  def df(self, dy: np.ndarray):
    return dy


class Mat(Layer):
  x: np.ndarray

  def __init__(self, size: int, activation: str = ''):
    super().__init__(activation)
    self.shape = (size,)

  def __call__(self, i: _Layer):
    if len(i.shape) != 1:
      raise ShapeIncompatible(f"prev layer shape: {i.shape}")
    return super().__call__(i)

  def initialize(self):
    if self.prev is None:
      raise NetworkNotReady()

    # Xavier Initialization
    self._w = np.random.randn(*self.prev.shape, *self.shape) / np.sqrt(self.prev.shape[0])
    self._dw = np.random.randn(*self.prev.shape, *self.shape) / np.sqrt(self.prev.shape[0])  # Xavier Initialization

  def f(self, x: np.ndarray, training: bool) -> np.ndarray:
    self.x = x
    return x.dot(self._w)

  def df(self, dy: np.ndarray) -> np.ndarray:
    self._dw = self.x.T.dot(dy)  # dy/dw
    return self._w.T.dot(dy)  # dy/dx


class Dense(Layer):
  x: np.ndarray

  def __init__(self, w_size: int, activation: str = ''):
    super().__init__(activation)
    self.shape = (w_size,)

  def __call__(self, i: _Layer):
    if len(i.shape) != 1:
      raise ShapeIncompatible(f"prev layer shape: {i.shape}")
    return super().__call__(i)

  def initialize(self):
    A = np.random.randn(*self.prev.shape, *self.shape) / np.sqrt(self.prev.shape[0])  # He Normal Initialization
    b = np.random.randn(1, *self.shape) / np.sqrt(self.prev.shape[0])  # Xavier Initialization
    self._w = np.concatenate((A, b))
    self._dw = np.random.randn(*self._w.shape) / np.sqrt(self.prev.shape[0])  # Xavier Initialization

  def f(self, x: np.ndarray, training: bool) -> np.ndarray:
    self.x = x
    return x.dot(self._w[:-1]) + self._w[-1]

  def df(self, dy: np.ndarray):
    db = dy.sum(axis=0).reshape((1, -1))
    da = self.x.T.dot(dy)
    self._dw = np.concatenate((da, db))  # dL/dw

    self.updater(self._w, self._dw)
    return dy.dot(da.T)  # dL/dx


class Dropout(_Layer):
  mask: np.ndarray

  def __init__(self, prob):
    super().__init__()
    self.prob = prob

  def __call__(self, i: Layer):
    self.shape = i.shape
    return super().__call__(i)

  def f(self, x: np.ndarray, training: bool) -> np.ndarray:
    if training:
      self.mask = np.random.rand(*self.shape) > self.prob
      return x * self.mask
    return x * self.prob

  def df(self, dy: np.ndarray):
    return dy * self.mask

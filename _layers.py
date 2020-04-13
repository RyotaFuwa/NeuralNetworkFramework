import copy
from abc import ABC, abstractmethod
from typing import Tuple, Union
import numpy as np
from Error import ShapeIncompatible

TYPE_SHAPE = Union[int, Tuple[int]]

"""Design Assumption:
 Shape of each layer is the shape of one sample (i.e it represents shape of one sample data, not mini-batch data).
Designer should implement layers so that it can take (batch_size, *previous_layer.shape),
and return (batch_size, *layer.shape)
"""


# Layer's fields shouldn't be accessed directly from user
class Layer(ABC, object):
  ONLY_IN_TRAINING: bool  # if True, the layer has to return an output with the same shape of the input
  LEARNABLE: bool  # if True, it has weights in side to be learned

  _shape: Tuple[int]  # shape of the layer (i.e the shape of output)
  _w: np.ndarray  # learnable weights
  _dw: np.ndarray  # dL/dw (L: loss function) It must have the same shape of _w
  _x: np.ndarray  # input from last layers
  _prev: Tuple['Layer']
  _next: Tuple['Layer']

  @property
  def shape(self):
    pass
  @shape.getter
  def shape(self):
    return self._shape

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

  @property
  def prev(self):
    pass
  @prev.getter
  def prev(self):
    return self._prev

  @property
  def next(self):
    pass
  @next.getter
  def next(self):
    return self._next

  def __str__(self, **kwargs):
    class_txt = self.__class__.__name__.ljust(5)
    w_txt = ":: {}".format((None, *self.shape))  # None changes depending on the batch_size
    param_txt = "{" + ', '.join(["{}: {}".format(k, v) for k, v in kwargs]) + "}"
    return class_txt + w_txt + param_txt

  @abstractmethod
  def __call__(self, i):
    """setup layer"""

  @abstractmethod
  def f(self, x: np.ndarray):
    """calculate output for x"""

  @abstractmethod
  def df(self, y: np.ndarray):
    """calculate dy/dx and return it. Also, store dy/dw as self.dw if the layer is learnable"""


# Input Layer. Define the input of the model
class Input(Layer):
  ONLY_IN_TRAINING = True
  LEARNABLE = False

  def __init__(self, shape: TYPE_SHAPE):
    self._prev = ()
    if type(shape) == int:
      self._shape = (shape,)
    else:
      self._shape = shape
    self._w = np.array([])
    self._dw = np.array([])

  def __call__(self, i):
    self._next = (i,)

  def f(self, x: np.ndarray):
    return x

  def df(self, dy: np.ndarray):
    pass


class SequenceLayer(Layer, ABC):
  ONLY_IN_TRAINING = False
  LEARNABLE = True

  def __init__(self):
    self._next = ()

  def __call__(self, i: Layer):
    self._prev = (i,)
    i._next = (self,)
    return self


class Mat(SequenceLayer):
  def __init__(self, w_size: int):
    self._shape = (w_size,)
    super().__init__()

  def __call__(self, i: Layer):
    if len(i._shape) != 1:  # check if the input shape is compatible
      raise ShapeIncompatible("Forward Layer accepts only 1-dim data as input")
    self._w = np.random.randn(*i._shape, *self._shape) * 0.001  # initialization of weights
    self._dw = np.zeros_like(self._w)
    SequenceLayer.__call__(self, i)
    return self

  def f(self, x: np.ndarray):
    self._x = np.array([x])
    return x.dot(self._w)

  def df(self, dy: np.ndarray):
    self._dw = self._prev[0]._w.T.dot(dy)  # dy/dw
    return self._w.T.dot(dy)  # dy/dx


# Dropout Layer.
class Dropout(SequenceLayer):
  ONLY_IN_TRAINING = True
  LEARNABLE = False
  _filter: np.ndarray

  def __init__(self, prob):
    self.prob = prob
    self._w = np.array([])
    self._dw = np.array([])
    super().__init__()

  def __call__(self, i: Layer):
    self._shape = i.shape
    self._filter = np.ones((1, *i.shape))
    super().__call__(i)
    return self

  def f(self, x: np.ndarray):
    self._x = x
    self._filter = np.where(np.random.rand(*self._filter.shape) > self.prob, 1.0, 0.0)
    return x * self._filter

  def df(self, dy: np.ndarray):
    return dy * self._filter


class AggregateLayer(Layer, ABC):  # TODO
  def __call__(self, *i: Layer):
    self._prev = i
    return self

import copy
from abc import ABC, abstractmethod
from typing import Tuple, Union
import numpy as np
from optimizers import Updater

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

  updater: Updater

  _shape: Tuple[int]  # shape of the layer (i.e the shape of output of its layer for one sample not for mini-batch data)
  _w: np.ndarray  # learnable weights
  _dw: np.ndarray  # dL/dw (L: loss function) It must have the same shape of _w
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

  @abstractmethod
  def __call__(self, i):
    """setup layer"""

  @abstractmethod
  def f(self, x: np.ndarray):
    """calculate output for x"""

  @abstractmethod
  def df(self, y: np.ndarray):
    """return dy/dx, store dy/dw to dw, and update w based on the derivative. if the layer is learnable"""

  def __str__(self, **kwargs):
    class_txt = self.__class__.__name__.ljust(5)
    w_txt = ":: {}".format((None, *self.shape))  # None changes depending on the batch_size
    param_txt = "{" + ', '.join(["{}: {}".format(k, v) for k, v in kwargs]) + "}"
    return class_txt + w_txt + param_txt


class SequenceLayer(Layer, ABC):
  ONLY_IN_TRAINING = False
  LEARNABLE = True

  def __init__(self):
    self._prev = None
    self._next = None
    self._w = None
    self._dw = None

  def __call__(self, i: Layer):
    self._prev = i
    i._next = self
    return self



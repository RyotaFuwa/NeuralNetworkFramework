import copy
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Set
import numpy as np
from Error import ShapeIncompatible
from activations import Activation

"""Design Assumption
each layer's shape is the shape of a sample.
designer should implement layers so that it can take (batch_size, *previous_layer.shape), and
return (batch_size, *layer.shape)
"""


# Layer's fields shouldn't be accessed directly from user
class Layer(ABC):
  CONST: Set[str] = {'shape', 'w'}  # accessible but not be modified
  ONLY_IN_TRAINING: bool

  shape: Tuple[int]  # shape of the layer (i.e the shape of output)
  w: np.ndarray  # learnable weights
  previous: Tuple['Layer']

  def __setattr__(self, key, value):
    if not (key in self.CONST):
      self.__dict__[key] = value

  def __str__(self):
    class_txt = self.__class__.__name__.ljust(15)

  def __eq__(self, other):
    return self.shape == other.shape

  def __ne__(self, other):
    return not self.__eq__(other)

  @abstractmethod
  def calculate(self, x: np.ndarray):
    """calculate output for x"""


# Input Layer. Define the input of the model
class Input(Layer):
  ONLY_IN_TRAINING = True
  LEARNABLE = False

  def __init__(self, shape: Tuple):
    self.previous = (None,)
    self.next = (None,)
    self.shape = shape
    self.W = np.array([])

  def __call__(self, x):
    pass

  def calculate(self, x: np.ndarray):
    return x


class SequenceLayer(Layer, ABC):
  def __call__(self, i: Layer):
    self.previous = (i,)
    if i.next == (None,):
      i.next = (self, )
    else:
      i.next = (*i.next, self)

  def create_str(self, shape='()', **kwargs):
    class_txt = self.__class__.__name__.ljust(15)
    w_txt = ":: {}".format((None, *shape))  # None changes depending on the batch_size
    param_txt = "{" + ', '.join(["{}: {}".format(k, v) for k, v in kwargs]) + "}"
    return class_txt + w_txt + param_txt


class AggregateLayer(Layer, ABC):
  def __call__(self, *i: Layer):
    self.previous = i


# Forward Layer. Take 1-dim input and return 1-dim output
class Forward(SequenceLayer):
  SequenceLayer.CONST.add('activation')
  ONLY_IN_TRAINING = False

  _A: np.ndarray
  _b: np.ndarray
  activation: Activation

  def __init__(self, w_size: int, activation: Activation):
    self.shape = (w_size,)
    self.activation = activation

  def __call__(self, i: Layer):
    if len(i.shape) != 1:  # check if the input shape is compatible
      raise ShapeIncompatible("Forward Layer accepts only 1-dim data as input")
    self._A = np.random.randn(*i.shape, *self.shape) * 0.001  # initialization of weights
    self._b = np.random.randn(1, *self.shape) * 0.001  # initialization of weights
    self.W = np.array([self._A, self._b])
    SequenceLayer.__call__(self, i)
    return self

  def __str__(self):
    return self.create_str(w=self.shape, activation=self.activation)

  def calculate(self, X: np.ndarray):
    return X.dot(self._A) + self._b


# Dropout Layer.
class Dropout(SequenceLayer):
  ONLY_IN_TRAINING = True
  _filter: np.ndarray

  def __init__(self, prob):
    self.prob = prob
    self.W = np.ndarray([])

  def __call__(self, i: Layer):
    self.shape = i.shape
    self._filter = np.ones((1, *i.shape))
    SequenceLayer.__call__(self, i)
    return self

  def __str__(self):
    return self.create_str()

  def calculate(self, x: np.ndarray):
    self._filter = np.where(np.random.rand(*self._fileter.shape) > self.prob, 1.0, 0.0)
    return x * self._filter


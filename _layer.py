from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional
import numpy as np

from optimizers import Updater


"""Design Assumption:
 Shape of each layer is the shape of one sample (i.e it represents shape of one sample data, not mini-batch data).
Designer should implement layers so that it can take (batch_size, *previous_layer.shape),
and return (batch_size, *layer.shape)
"""


class _Layer(ABC):
  shape: Tuple[int]  # shape of the layer (i.e the shape of the output from this layer => [# of sample, *shape])
  prev: Optional['_Layer']
  next: Optional['_Layer']

  @abstractmethod
  def __init__(self):
    self.shape = (0,)
    self.prev = None
    self.next = None

  @abstractmethod
  def f(self, x: np.ndarray, training: bool) -> np.ndarray:
    """forward calculation"""

  @abstractmethod
  def df(self, x: np.ndarray) -> np.ndarray:
    """back propagation"""

  def __call__(self, i: '_Layer') -> '_Layer':
    self.prev = i
    i.next = self
    return self.tail()

  def __str__(self, **kwargs):
    class_txt = self.__class__.__name__.ljust(5)
    w_txt = ":: {}".format((None, *self.shape))  # None changes depending on the batch_size
    param_txt = "{" + ', '.join(["{}: {}".format(k, v) for k, v in kwargs]) + "}"
    return class_txt + w_txt + param_txt

  def head(self):
    current = self
    while current.prev is not None:
      current = current.prev
    return current

  def tail(self):
    current = self
    while current.next is not None:
      current = current.next
    return current



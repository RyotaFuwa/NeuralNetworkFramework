from copy import deepcopy
from typing import Union, Tuple, List
import numpy as np

# typing definition
TYPE_INPUT = Union['Input', Tuple['Input']]
TYPE_LAYER = Union['Layer', Tuple['Layer']]


class W(object):
  w: List[np.ndarray] = []

  def __init__(self, other):
    self.w = deepcopy(other.w)

  def __iadd__(self, other):
    for i, j in zip(self.w, other.w):
      i += j

  def __imul__(self, other):
    for i, j in zip(self.w, other.w):
      i *= j

  def __isub(self, other):
    for i, j in zip(self.w, other.w):
      i -= j

  def __idiv(self, other):
    for i, j in zip(self.w, other.w):
      i /= j

  def append(self, w: np.ndarary):
    self.w_list.append(w)


def get_I(x: np.ndarray, batch_size: int, shuffle: bool = False) -> np.ndarray:
  if shuffle:
    index = np.random.randint(0, x.shape[0], size=(batch_size,))
  else:
    index = np.arange(batch_size)
  return x[index, :]


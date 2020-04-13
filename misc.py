from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union, Tuple, List
import numpy as np
from _layers import Layer

# typing definition


class W(object):
  params: Tuple[np.ndarray]

  def __init__(self, w, deep_copy=False):
    if deep_copy:
      self.params = deepcopy(w)
    else:
      self.params = w

  def __iadd__(self, other):
    for i, j in zip(self.params, other.params):
      i += j

  def __imul__(self, other):
    for i, j in zip(self.params, other.params):
      i *= j

  def __isub__(self, other):
    for i, j in zip(self.params, other.params):
      i -= j

  def __idiv__(self, other):
    for i, j in zip(self.params, other.params):
      i /= j

  def append(self, w_in: np.ndarray):
    self.w.append(w_in)



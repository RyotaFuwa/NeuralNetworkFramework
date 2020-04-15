from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union, Tuple, List
import numpy as np


def random_like(w: np.ndarray, mue=0.0, sigma=1.0):
  def helper(w: np.ndarray, place: list):
    if w.dtype == 'O':
      new_place = []
      for obj in w:
        helper(obj, new_place)
      place.append(np.array(new_place))
    else:
      value = (np.random.randn(*w.shape) - mue) * sigma
      place.append(value)
  out = []
  helper(w, out)
  return np.array(out[0])


def debug(a: np.ndarray, b: np.ndarray):
  return np.all(a == b)


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



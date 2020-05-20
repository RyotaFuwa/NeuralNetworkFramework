from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union, Tuple, List
import numpy as np


def normalize(x: np.ndarray):
  mean = x.mean()
  scale = np.max(np.abs(x - mean))
  return (x - mean) / scale


def to_one_hot(x):
  if x.shape[-1] == 1 and len(x.shape) > 1:
    shape = x.shape[:-1]
  else:
    shape = (x.shape[0],)
  x = x.flatten()
  num_classes = int(np.max(x) + 1)
  one_hot = np.zeros((x.shape[0], num_classes))
  one_hot[np.arange(x.shape[0]), x] = 1
  return one_hot.reshape(shape + (num_classes,))


def split_data(nparray, ratio=0.2, random=True):
  num_of_sample = nparray.shape[0]
  separate_point = int(num_of_sample * ratio)
  indices = np.arange(num_of_sample)
  if random:
    np.random.shuffle(indices)
  return nparray[indices[:separate_point]], nparray[separate_point:]



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



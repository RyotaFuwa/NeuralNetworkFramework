from abc import ABC, abstractmethod
import numpy as np
from misc.constants import EPSILON


class Loss(ABC):
  """loss function. x and y must have a same shape"""
  x: np.ndarray  # output from network
  y: np.ndarray  # label

  @staticmethod
  @abstractmethod
  def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """calculate loss value"""

  @staticmethod
  @abstractmethod
  def df(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """calculate dL/dy"""


class MSE(Loss):

  @staticmethod
  def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.mean(np.square(x - y))

  @staticmethod
  def df(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    batch_size = y.shape[0]
    num_of_weights = y.shape[1]
    return 2.0 * (x - y) / num_of_weights / batch_size


class CrossEntropy(Loss):

  @staticmethod
  def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.mean(-np.sum(y * np.log(x + EPSILON), axis=-1))

  @staticmethod
  def df(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    batch_size = y.shape[0]
    return - y / (x + EPSILON) / batch_size


class SparseCrossEntropy(Loss):
  """assume y has integers between 0 and # of categories - 1
  for shape of x: (# of samples, # of categories)
  """

  @staticmethod
  def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """y has shape of (# of samples, ) or (# of samples, 1)"""
    if y.ndim == 2:
      y = y.reshape((-1))
    x = x[np.arange(x.shape[0]), y]
    return np.mean(np.log(x + EPSILON))

  @staticmethod
  def df(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """y has shape of (# of samples, ) or (# of samples, 1)"""
    if y.ndim == 2:
      y = y.reshape((-1))
    mask = np.zeros_like(x)
    mask[np.arange(mask.shape[0]), y] = 1
    batch_size = y.shape[0]
    return - mask / (x + EPSILON) / batch_size


class SoftmaxWithCrossEntropy(Loss):

  @staticmethod
  def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """this function is not supposed to be used"""
    pass

  @staticmethod
  def df(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    batch_size = y.shape[0]
    return (x - y) / batch_size


REGISTERED_LOSSES = {
  'mse': MSE,
  'cross_entropy': CrossEntropy,
  'sparse_cross_entropy': SparseCrossEntropy,
}


def loss_load(key: str = ''):
  if key in REGISTERED_LOSSES:
    return REGISTERED_LOSSES[key]
  else:
    return None


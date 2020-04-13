from abc import ABC, abstractmethod
import numpy as np
from Error import ShapeIncompatible


class Loss(ABC):
  x: np.ndarray  # output from network
  y: np.ndarray  # label
  batch_size: int
  @abstractmethod
  def f(self, x: np.ndarray, y: np.ndarray):
    """

    :param x:
    :param y:
    :return:
    """

  @abstractmethod
  def df(self):
    """

    :return:
    """


class MSE(Loss):
  def f(self, x: np.ndarray, y: np.ndarray):
    self.x = x
    self.y = y
    self.batch_size = x.shape[0]

    if x.shape != y.shape:
      raise ShapeIncompatible("x and y's sizes are different")
    return np.mean(np.square(x - y))

  def df(self):
    return 2.0/sum(self.shape[1:]) * (self.x - self.y)


class CrossEntropy(Loss):
  def f(self, x: np.ndarray, y: np.ndarray):
    if x.shape != y.shape:
      raise ShapeIncompatible("Size Different For (x: {}, y: {})".format(x.shape, y.shape))
    return np.mean(np.sum(y * np.log(x), axis=1))

  def df(self):
    return - self.y / self.x





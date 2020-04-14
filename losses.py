from abc import ABC, abstractmethod
import numpy as np
from Error import ShapeIncompatible


class Loss(ABC):
  x: np.ndarray  # output from network
  y: np.ndarray  # label
  loss: np.float32
  @abstractmethod
  def f(self, x: np.ndarray, y: np.ndarray):
    if x.shape != y.shape:
      raise ShapeIncompatible("Size Different For (x: {}, y: {})".format(x.shape, y.shape))
    self.x = x
    self.y = y
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
    super().f(x, y)

    if x.shape != y.shape:
      raise ShapeIncompatible("x and y's sizes are different")
    self.loss = np.mean(np.square(x - y))
    return self.loss

  def df(self):
    batch_size = self.y.shape[0]
    data_shape = self.y.shape[1:]
    return 2.0/sum(data_shape) * (self.x - self.y) / batch_size


class CrossEntropy(Loss):
  def f(self, x: np.ndarray, y: np.ndarray):
    super().f(x, y)
    return np.mean(-np.sum(y * np.log(x), axis=1))

  def df(self):
    batch_size = self.y.shape[0]
    return - self.y / self.x / batch_size





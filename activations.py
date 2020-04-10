from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
  @staticmethod
  def f(x: np.ndarray) -> np.ndarray:
    """activation function"""
  @staticmethod
  def df(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """derivative of activation function"""


class Sigmoid(Activation):
  @staticmethod
  def f(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))
  @staticmethod
  def df(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return y * (1 - y)


class ReLU(Activation):
  @staticmethod
  def f(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, 0)
  @staticmethod
  def df(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0)


class Tanh(Activation):
  @staticmethod
  def f(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)
  @staticmethod
  def df(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 1 / np.square(np.cosh(x))


class Softmax(Activation):
  @staticmethod
  def f(x: np.ndarray) -> np.ndarray:
    max_v = np.max(x)
    return np.exp(x - max_v) / np.sum(np.exp(x - max_v))
  @staticmethod
  def df(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.ones_like(x)


class Linear(Activation):
  @staticmethod
  def f(x: np.ndarray) -> np.ndarray:
    return x
  @staticmethod
  def df(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.ones_like(y)


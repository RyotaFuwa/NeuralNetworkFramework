from abc import ABC, abstractmethod
import numpy as np

EPSILON = np.finfo(float).eps


class Generator(ABC):
  CLASSIFIER_TYPE: str
  CLASSES: list = []

  @staticmethod
  @abstractmethod
  def generate(X: np.ndarray, *args, **kwargs) -> np.ndarray:
    """

    :param X: np.ndarray with shape of (# of sample, 2) where each sample holds x, y coordinate
    :param args: optional
    :param kwargs: optional
    :return: np.ndarray with shape of (# of sample, 1) where the value represents "a feature" of the sample
    """


class HurricaneGenerator(Generator):
  CLASSIFIER_TYPE = 'median'

  @staticmethod
  def generate(X: np.ndarray, twist_coef=2, *args, **kwargs) -> np.ndarray:
    x = X[:, 0]
    y = X[:, 1]
    theta = np.arctan(y / (x + EPSILON)) + np.pi / 2.0
    theta += + np.where(x < 0, np.pi, 0)
    radius = np.sqrt(np.square(x) + np.square(y))
    return (np.power(theta, twist_coef) * radius).reshape((-1, 1))


class CircleGenerator(Generator):
  CLASSIFIER_TYPE = 'median'

  @staticmethod
  def generate(X: np.ndarray, steepness=1.0, *args, **kwargs) -> np.ndarray:
    x = X[:, 0]
    y = X[:, 1]
    radius = np.sqrt(np.square(x) + np.square(y))
    return (radius * steepness).reshape((-1, 1))


class CheckerGenerator(Generator):
  """Assume num_of_class is always set to 2"""
  CLASSIFIER_TYPE = 'specified'
  CLASSES = [0.0]

  @staticmethod
  def generate(X: np.ndarray, *args, **kwargs) -> np.ndarray:
    num_of_samples = X.shape[0]
    Y = np.zeros((num_of_samples, 1))

    for idx in range(num_of_samples):
      if X[idx, 0] > 0:
        if X[idx, 1] > 0:
          Y[idx, 0] = 1
        else:
          Y[idx, 0] = -1
      else:
        if X[idx, 1] > 0:
          Y[idx, 0] = -1
        else:
          Y[idx, 0] = 1
    return Y

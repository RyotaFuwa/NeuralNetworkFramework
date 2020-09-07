from typing import Type
import numpy as np
from .generators import Generator

"""2D classification data generator"""


def classifier(Y: np.ndarray, type: str = 'median', num_of_class: int = 3, classes=[]):
  """
  :param Y: np.ndarray with shape (# of sample)
  :param type: classifier type, ['median', 'mean', 'specified']
  :param num_of_class:  if type is median or mean, num_of_class will be used.
  :param classes: if type is specified, then this classes parameter will be used.
  :return: np.ndarray with shape (# of sample) filled with int representing class
  """
  if type == 'median':
    if num_of_class is 0:
      assert 'num_of_class can\'t be 0'
    ratios = np.array([100 / num_of_class * i for i in range(1, num_of_class)])
    classes = np.percentile(Y, ratios, axis=0)
  elif type == 'specified':
    if len(classes) == 0:
      assert 'classes is not given'
  else:
    assert 'invalid classifier type'

  for idx in range(Y.shape[0]):
    for i, threshold in enumerate(classes):
      if Y[idx, 0] < threshold:
        Y[idx, 0] = i
        break
    else:
      Y[idx, 0] = len(classes)
  return Y.astype('int8')


def generate_data(num_of_class: int, num_of_sample: int, generator: Type[Generator]):
  # limit of num_of_class is set to 100
  if num_of_class > 100:
    num_of_class = 100
  X = np.random.random((num_of_sample, 2)) - 0.5
  X /= np.abs(X).max()

  Y = generator.generate(X)

  return X, classifier(Y, type=generator.CLASSIFIER_TYPE, num_of_class=num_of_class, classes=generator.CLASSES)


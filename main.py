import numpy as np
import matplotlib.pyplot as plt

from demo.liner_regression import linear_regression
from demo.keras.linear_regression import linear_regression as linear_regression_keras

from demo.linear_classification import linear_classification
from demo.keras.linear_classification import linear_classification as linear_classification_keras

from demo.universal_approximation import universal_approximation
from demo.keras.universal_approximation import universal_approximation as universal_approximation_keras

from demo.binary_classification import binary_classification
from demo.keras.binary_classification import binary_classification as binary_classification_keras


def main():
  """Linear Regression"""
  linear_params = {'a': np.random.randn(), 'b': np.random.randn() * 10}
  # linear_regression(**linear_params)
  # linear_regression_keras(**linear_params)

  """Linear Classification"""
  # linear_classification(**linear_params)
  # linear_classification_keras(**linear_params)

  """Universal Approximation"""
  X = np.linspace(-3.14, 3.14, 500)
  X = X.reshape((-1, 1))
  f = np.sin
  plt.plot(X, f(X))
  plt.show()
  universal_approximation(f, X)
  # universal_approximation_keras(f, X)

  """Binary Classification"""
  # binary_classification()
  # binary_classification_keras()


if __name__ == '__main__':
  main()

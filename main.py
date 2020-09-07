import numpy as np
import matplotlib.pyplot as plt
from misc import measure_time

from demo.liner_regression import linear_regression
from demo.keras.linear_regression import linear_regression as linear_regression_keras

from demo.linear_classification import linear_classification

from demo.universal_approximation import universal_approximation
from demo.keras.universal_approximation import universal_approximation as universal_approximation_keras

from demo.binary_classification import binary_classification
from demo.keras.binary_classification import binary_classification as binary_classification_keras


def main():
  # linear regression
  linear_params = {'a': 1.0, 'b': 0.0}
  # linear_regression(**linear_params)
  # linear_regression_keras(**linear_params)

  # linear classification
  linear_classification(**linear_params)

  # universal_approximation
  # X = np.linspace(-3.14, 3.14, 200)
  # X = X.reshape((-1, 1))
  # f = np.sin
  # plt.plot(X, f(X))
  # plt.show()
  # universal_approximation(f, X)
  # universal_approximation_keras(f, X)

  # binary_classification()
  # binary_classification_keras()


if __name__ == '__main__':
  main()

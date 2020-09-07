import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD

from misc import split_data


def linear_regression(a=1.0, b=0.0):
  X = np.linspace(-100, 100, 200)
  X /= X.max()
  X = X.reshape((-1, 1))
  [train_x, test_x] = split_data(X, ratio=0.8, random=True)
  train_y = a * train_x + b
  test_y = a * test_x + b

  # build simple FNN
  i = Input(1)
  x = Dense(1)(i)

  # define trainer
  # create model
  model = Model(i, x)

  # training process
  model.compile(optimizer='adam', loss='mse')
  model.fit(train_x, train_y, batch_size=50, epochs=50)

  # predict
  y_hat = model.predict(test_x)
  plt.plot(test_x, test_y, 'b')
  plt.plot(test_x, y_hat, 'r')
  plt.show()

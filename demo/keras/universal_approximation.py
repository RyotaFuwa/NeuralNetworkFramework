import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from misc import split_data


def universal_approximation(f, x):
  [train_x, test_x] = split_data(x, ratio=0.8, random=True)
  train_y = np.sin(train_x)
  test_y = np.sin(test_x)

  # build simple FNN
  model = Sequential()
  model.add(Dense(30, input_shape=(1,), activation='relu'))
  model.add(Dense(1))

  model.compile(loss='mse', optimizer='sgd')

  # training process
  model.fit(train_x, train_y, batch_size=50, epochs=500)

  # predict
  y_hat = model.predict(test_x)
  plt.plot(test_x, test_y, 'bo')
  plt.plot(test_x, y_hat, 'ro')
  plt.show()


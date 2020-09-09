import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from misc.utils import split_data


def universal_approximation(f, x):
  [train_x, test_x] = split_data(x, ratio=0.75, random=True)
  train_y = np.sin(train_x)
  test_x = np.sort(test_x, axis=0)
  test_y = f(test_x)

  # build simple FNN
  model = Sequential()
  model.add(Dense(50, input_shape=(1,), activation='relu'))
  model.add(Dense(1))

  model.compile(loss='mse', optimizer='adam')

  # training process
  model.fit(train_x, train_y, batch_size=100, epochs=1000)
  layer = model.get_layer(index=0)

  plt.plot(model.history.history['loss'])
  plt.show()

  # predict
  y_hat = model.predict(test_x)
  plt.plot(test_x, test_y, 'b-', label='original')
  plt.plot(test_x, y_hat, 'r-', label='predicted')
  plt.legend()
  plt.show()

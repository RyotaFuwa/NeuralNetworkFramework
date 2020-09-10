import numpy as np
import matplotlib.pyplot as plt
from Trainer import Trainer
from layers import Input, Dense
from misc.utils import split_data
from models import Sequential
from optimizers.optimizers import Adam


def linear_regression(a=1.0, b=0.0):
  X = np.linspace(-100, 100, 200)
  X = X.reshape((-1, 1))
  [train_x, test_x] = split_data(X, ratio=0.8, random=True)
  train_y = a * train_x + b
  test_y = a * test_x + b

  i = Input(1)
  x = Dense(1)(i)

  # define trainer
  trainer = Trainer(loss='mse', optimizer=Adam(learning_rate=0.2), batch_size=50, epochs=50)

  # create model
  model = Sequential(i, x, trainer)

  model.summary()

  # training process
  model.fit(train_x, train_y)

  # predict
  y_hat = model.predict(test_x)
  plt.plot(test_x, test_y, 'b')
  plt.plot(test_x, y_hat, 'r')
  plt.show()



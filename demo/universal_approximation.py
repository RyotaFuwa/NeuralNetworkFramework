import time
import numpy as np
import matplotlib.pyplot as plt
from Trainer import Trainer
from layers import Input, Dense
from misc.utils import split_data
from models import Sequential
from optimizers import SGD, Adam


def universal_approximation(f, x):
  [train_x, test_x] = split_data(x, ratio=0.8, random=True)
  train_y = f(train_x)

  test_x = np.sort(test_x, axis=0)
  test_y = f(test_x)

  # build simple FNN
  i = Input(1)
  x = Dense(50, activation='relu')(i)
  x = Dense(1)(x)

  # define trainer
  trainer = Trainer(loss='mse', optimizer=Adam(learning_rate=0.01), batch_size=50, epochs=750)

  # create model
  model = Sequential(i, x, trainer)

  model.summary()

  # training process
  start = time.time()
  model.fit(train_x, train_y)
  print(time.time() - start)

  plt.plot(range(len(model.history['loss'])), model.history['loss'])
  plt.show()

  # predict
  y_hat = model.predict(test_x)
  plt.plot(test_x, test_y, 'b-', label='original')
  plt.plot(test_x, y_hat, 'r-', label='predicted')
  plt.legend()
  plt.show()



import numpy as np
import matplotlib.pyplot as plt
from Trainer import Trainer
from activations import ReLU, Sigmoid, Tanh
from layers import Input, Forward
from losses import MSE
from misc import split_data
from models import Sequential
from optimizers import SGD, Adam


def universal_approximation(f, x):
  [train_x, test_x] = split_data(x, 0.8, True)
  train_y = np.sin(train_x)
  test_y = np.sin(test_x)

  # build simple FNN
  i = Input(1)
  x = Forward(30, activation=ReLU())(i)
  x = Forward(1)(x)

  # define trainer
  trainer = Trainer(loss=MSE(), optimizer=Adam(), batch_size=50, epochs=50)

  # create model
  model = Sequential(i, x, trainer)

  # training process
  model.train(train_x, train_y)

  plt.plot(range(len(model.history['loss'])), model.history['loss'])
  plt.show()

  # predict
  y_hat = model.predict(test_x)
  plt.plot(test_x, test_y, 'bo')
  plt.plot(test_x, y_hat, 'ro')
  plt.show()



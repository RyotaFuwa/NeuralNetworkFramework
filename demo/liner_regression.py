import numpy as np
import matplotlib.pyplot as plt
from Trainer import Trainer
from layers import Input, Forward
from activations import ReLU
from losses import MSE
from misc import split_data
from models import Sequential
from optimizers import SGD, Adam


def linear_regression(a=1.0, b=0.0):
  X = np.linspace(-100, 100, 200)
  X /= X.max()
  X = X.reshape((-1, 1))
  [train_x, test_x] = split_data(X, ratio=0.8, random=True)
  train_y = a * train_x + b
  test_y = a * test_x + b

  i = Input(1)
  x = Forward(1, activation=ReLU())(i)
  x = Forward(1)(x)

  # define trainer
  trainer = Trainer(loss=MSE(), optimizer=Adam(0.1), batch_size=50, epochs=50)

  # create model
  model = Sequential(i, x, trainer)

  print(model)

  # training process
  model.train(train_x, train_y)

  plt.plot(range(len(model.history['loss'])), model.history['loss'])
  plt.show()

  # predict
  y_hat = model.predict(test_x)
  plt.plot(test_x, test_y, 'b')
  plt.plot(test_x, y_hat, 'r')
  plt.show()



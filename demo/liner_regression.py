import numpy as np
import matplotlib.pyplot as plt
from Trainer import Trainer
from layers import Input, Forward
from losses import MSE
from misc import split_data
from models import FNN
from optimizers import SGD


def linear_regression(a=1.0, b=0.0):
  X = np.linspace(-10, 10, 1000).reshape((-1, 1))
  [train_x, test_x] = split_data(X, 0.8, True)
  train_y = a * train_x + b
  test_y = a * test_x + b

  # build simple FNN
  i = Input(1)
  x = Forward(1)(i)

  # define trainer
  trainer = Trainer(loss=MSE(), optimizer=SGD(0.01), batch_size=50, epochs=50)

  # create model
  model = FNN(i, x, trainer)

  # training process
  model.train(train_x, train_y)

  # predict
  y_hat = model.predict(test_x)
  plt.plot(test_x, test_y, 'b')
  plt.plot(test_x, y_hat, 'r')
  plt.show()



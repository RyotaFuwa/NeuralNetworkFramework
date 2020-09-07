import numpy as np
import matplotlib.pyplot as plt
from Trainer import Trainer
from layers import Input, Forward
from activations import Softmax
from losses import CrossEntropy
from misc import split_data, to_one_hot
from models import Sequential
from optimizers import SGD, Adam


def simple_plot(x, y, a, b):
  color = {0: 'b', 1: 'r'}
  if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
  y = [color[i] for i in y]
  plt.scatter(x[:, 0], x[:, 1], c=y, s=2.5)
  plt.plot(x[:, 0], a * x[:, 0] + b)
  plt.show()


def linear_classification(a=1.0, b=0.0):
  X = np.random.randn(500, 2)
  X /= X.max()
  [train_x, test_x] = split_data(X, ratio=0.8, random=True)
  train_y = to_one_hot(np.where(a * train_x[:, 0] + b > train_x[:, 1], 1, 0))
  test_y = np.where(a * test_x[:, 0] + b > test_x[:, 1], 1, 0)

  simple_plot(train_x, train_y, a, b)

  # build simple FNN
  i = Input(2)
  x = Forward(2, activation=Softmax())(i)

  # define trainer
  trainer = Trainer(loss=CrossEntropy(), optimizer=Adam(0.1), batch_size=50, epochs=50)

  # create model
  model = Sequential(i, x, trainer)

  print(model)

  # training process
  model.train(train_x, train_y)

  plt.plot(range(len(model.history['loss'])), model.history['loss'])
  plt.show()

  # predict
  y_hat = model.predict(test_x)
  y_hat = np.argmax(y_hat, axis=1)
  print("accuracy: {:.5f}".format(model.evaluate(y_hat, test_y)))
  simple_plot(test_x, y_hat, a, b)



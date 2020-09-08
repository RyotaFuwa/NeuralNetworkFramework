import time
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
  [train_x, test_x] = split_data(x, ratio=0.8, random=True)
  train_y = f(train_x)

  test_x = np.sort(test_x, axis=0)
  test_y = f(test_x)

  # build simple FNN
  i = Input(1)
  x = Forward(50, activation=ReLU())(i)
  x = Forward(1)(x)

  # define trainer
  trainer = Trainer(loss=MSE(), optimizer=Adam(learning_rate=0.01, clipvalue=1.5), batch_size=50, epochs=750)

  # create model
  model = Sequential(i, x, trainer)

  print(model)

  # training process
  start = time.time()
  model.train(train_x, train_y)
  print(time.time() - start)

  plt.plot(range(len(model.history['loss'])), model.history['loss'])
  plt.show()

  # predict
  y_hat = model.predict(test_x)
  plt.plot(test_x, test_y, 'b-', label='original')
  plt.plot(test_x, y_hat, 'r-', label='predicted')
  plt.legend()
  plt.show()



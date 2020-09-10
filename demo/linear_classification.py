import numpy as np
import matplotlib.pyplot as plt
from Trainer import Trainer
from layers import Input, Dense
from misc.utils import split_data, to_one_hot
from models import Sequential
from optimizers.optimizers import Adam


def simple_plot(x, y, a, b):
  color = {0: 'b', 1: 'r'}
  if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
  y = [color[i] for i in y]
  plt.scatter(x[:, 0], x[:, 1], c=y, s=2.5)
  plt.plot(x[:, 0], a * x[:, 0] + b)
  plt.show()


def linear_classification(a=1.0, b=0.0, graph=False):

  # prepare data
  x = np.linspace(-100, 100, 200)
  y = a * x + b
  X = np.array(list(zip(x, y))) + np.random.randn(200, 2) * 100
  Y = to_one_hot(np.where(a * X[:, 0] + b > X[:, 1], 1, 0))
  (train_x, train_y), (test_x, test_y) = split_data(X, Y, ratio=0.8, random=True)

  # build simple FNN
  i = Input(2)
  x = Dense(2, activation='softmax')(i)

  # define trainer
  trainer = Trainer(
    loss='cross_entropy',
    optimizer=Adam(learning_rate=0.05),
    batch_size=50,
    epochs=50,
    metrics=['accuracy']
  )

  # create model
  model = Sequential(i, x, trainer)

  model.summary()

  # training process
  model.fit(train_x, train_y)
  print(model.evaluate(test_x, test_y))

  if graph:
    plt.plot(model.history['loss'])
    plt.show()

    # predict
    y_hat = model.predict(test_x)
    y_hat = np.argmax(y_hat, axis=1)
    simple_plot(test_x, y_hat, a, b)

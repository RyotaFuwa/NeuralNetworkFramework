import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD, Adam

from misc.utils import split_data, to_one_hot


def simple_plot(x, y, a, b):
  color = {0: 'b', 1: 'r'}
  if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
  y = [color[i] for i in y]
  plt.scatter(x[:, 0], x[:, 1], c=y, s=2.5)
  plt.plot(x[:, 0], a * x[:, 0] + b)
  plt.show()


def linear_classification(a=1.0, b=0.0):
  x = np.linspace(-100, 100, 200)
  y = a * x + b
  X = np.array(list(zip(x, y))) + np.random.randn(200, 2) * 100
  Y = np.where(a * X[:, 0] + b > X[:, 1], 1, 0)
  (train_x, train_y), (test_x, test_y) = split_data(X, Y, ratio=0.8, random=True)
  train_y = to_one_hot(train_y)
  test_y = np.where(a * test_x[:, 0] + b > test_x[:, 1], 1, 0)


  # build simple FNN
  i = Input(2)
  x = Dense(2, activation='softmax')(i)

  # define trainer

  # create model
  model = Model(i, x)
  model.compile(optimizer=Adam(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])

  model.summary()

  # training process
  model.fit(train_x, train_y, batch_size=50, epochs=50)

  # predict
  y_hat = model.predict(test_x)
  y_hat = np.argmax(y_hat, axis=1)
  simple_plot(test_x, y_hat, a, b)

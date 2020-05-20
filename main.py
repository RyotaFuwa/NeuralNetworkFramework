import os
import numpy as np
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt

from misc import normalize, to_one_hot, split_data
from activations import ReLU, Softmax, Linear
from layers import Input, Dropout, Forward
from models import FNN
from Trainer import Trainer
from losses import CrossEntropy, SoftmaxWithCrossEntropy, MSE
from optimizers import SGD, Adam


def load_arff(path):
  data = arff.loadarff(path)
  data = pd.DataFrame(data[0]).to_numpy()
  return data


def simple_plot(x, y):
  Color = {0: 'b', 1: 'r'}
  y = np.argmax(y, axis=1)
  plt.scatter(x[:, 0], x[:, 1], c=y)
  plt.plot()
  plt.show()


def main():
  # simple example: binary classification of 2d float data
  data_dir = "data/examples"
  train_data_path = os.path.join(data_dir, 'training.arff')
  test_data_path = os.path.join(data_dir, 'test.arff')
  train_data = load_arff(train_data_path)
  test_data = load_arff(test_data_path)

  def separate_label(data):
    X = normalize(data[:, :2].astype('float32'))
    Y = np.where(data[:, 2] == b'black', 0, 1)
    return X, Y

  train_x, train_y = separate_label(train_data)
  train_y = to_one_hot(train_y)
  test_x, test_y = separate_label(test_data)
  test_y = to_one_hot(test_y)

  # simple example no. two
  train_x = np.random.randn(1000, 2)
  train_y = to_one_hot(np.where(train_x[:, 0] < train_x[:, 1], 1, 0))
  simple_plot(train_x, train_y)
  train_x, test_x = split_data(train_x)
  train_y, test_y = split_data(train_y)

  simple_plot(test_x, test_y)

  # build simple FNN
  i = Input(2)
  x = Forward(10)(i)
  x = Forward(2)(x)

  # define trainer
  trainer = Trainer(loss=MSE(), optimizer=Adam(0.5, 0.95, 0.97), batch_size=200, epochs=2000)

  # create model
  model = FNN(i, x, trainer)

  # training process
  model.train(train_x, train_y)

  # predict
  y_hat = model.predict(test_x)
  print(y_hat)
  simple_plot(test_x, y_hat)


if __name__ == '__main__':
  main()


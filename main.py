import os
import numpy as np
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
from activations import ReLU, Softmax, Linear
from layers import Input, Dropout, Forward
from models import FNN
from Trainer import Trainer
from losses import CrossEntropy, SoftmaxWithCrossEntropy
from optimizers import SGD, Adam


def load_arff(path):
  data = arff.loadarff(path)
  data = pd.DataFrame(data[0]).to_numpy()
  return data


def normalize(x: np.ndarray):
  mean = x.mean()
  scale = np.max(np.abs(x - mean))
  return (x - mean) / scale


def to_one_hot(x):
  if x.shape[-1] == 1 and len(x.shape) > 1:
    shape = x.shape[:-1]
  else:
    shape = (x.shape[0],)
  x = x.flatten()
  num_classes = int(np.max(x) + 1)
  one_hot = np.zeros((x.shape[0], num_classes))
  one_hot[np.arange(x.shape[0]), x] = 1
  return one_hot.reshape(shape + (num_classes,))


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

  # simple_plot(train_x, train_y)

  # build simple FNN
  i = Input(2)
  x = Forward(10, ReLU)(i)
  x = Forward(2)(x)

  # define trainer
  trainer = Trainer(loss=SoftmaxWithCrossEntropy(), optimizer=Adam(0.5, 0.95, 0.97), batch_size=200, epochs=2000)

  # create model
  model = FNN(i, x, trainer)

  # training process
  model.train(train_x, train_y)

  # predict
  model.predict(test_x)


if __name__ == '__main__':
  main()


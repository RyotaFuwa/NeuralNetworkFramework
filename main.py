import os
import numpy as np
import pandas as pd
from scipy.io import arff
from activations import ReLU, Softmax
from layers import Input, Dropout, Forward
from models import FNN
from Trainer import Trainer
from losses import CrossEntropy
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
  x = x.ravel()
  num_classes = int(np.max(x) + 1)
  one_hot = np.zeros((x.shape[0], num_classes))
  one_hot[np.arange(x.shape[0]), x] = 1
  return one_hot.reshape(shape + (num_classes,))


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
  test_x, test_y = separate_label(test_data)
  train_y = to_one_hot(train_y)
  test_y = to_one_hot(test_y)

  # build simple FNN
  i = Input(2)
  x = Forward(10, ReLU)(i)
  x = Dropout(0.2)(x)
  x = Forward(10, ReLU)(x)
  x = Forward(2, Softmax)(x)
  model = FNN(i, x)

  # training process
  trainer = Trainer(loss=CrossEntropy(), optimizer=Adam(0.9, 0.95, 0.97), batch_size=200, epochs=200)
  model.train(train_x, train_y, trainer)
  model.predict(test_x)


if __name__ == '__main__':
  main()


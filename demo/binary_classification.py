import os
import numpy as np
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt

from misc.utils import normalize, to_one_hot
from layers import Input, Dense
from models import Sequential
from Trainer import Trainer
from optimizers.optimizers import Adam


def load_arff(path):
  data = arff.loadarff(path)
  data = pd.DataFrame(data[0]).to_numpy()
  return data


def simple_plot(x, y):
  color = {0: 'b', 1: 'r'}
  y = np.argmax(y, axis=1)
  y = [color[i] for i in y]
  plt.scatter(x[:, 0], x[:, 1], c=y)
  plt.show()


# demo
def binary_classification():
  def separate_label(data):
    X = normalize(data[:, :2].astype('float32'))
    Y = np.where(data[:, 2] == b'black', 0, 1)
    return X, Y

  # prepare train data
  data_dir = "data/examples/binary_classification"
  train_data_path = os.path.join(data_dir, 'training.arff')
  train_data = load_arff(train_data_path)
  train_x, train_y = separate_label(train_data)
  train_y = to_one_hot(train_y)

  # build simple FNN
  i = Input(2)
  x = Dense(30, activation='relu')(i)
  x = Dense(30, activation='relu')(x)
  x = Dense(2, activation='softmax')(x)

  # define trainer
  trainer = Trainer(loss='cross_entropy', optimizer=Adam(clipvalue=1.0), batch_size=256, epochs=500, metrics=['accuracy'])

  # create model
  model = Sequential(i, x, trainer)

  model.summary()

  # training process
  model.fit(train_x, train_y)

  plt.plot(range(len(model.history['loss'])), model.history['loss'])
  plt.show()

  # predict
  test_data_path = os.path.join(data_dir, 'test.arff')
  test_data = load_arff(test_data_path)
  test_x, _ = separate_label(test_data)

  y_hat = model.predict(test_x)
  simple_plot(test_x, y_hat)

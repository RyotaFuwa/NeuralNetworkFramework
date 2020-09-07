import os
import numpy as np
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from misc import normalize, to_one_hot

# for mac
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def load_arff(path):
  data = arff.loadarff(path)
  data = pd.DataFrame(data[0])

  # normalization
  train_x = data.loc[:, ['x', 'y']].to_numpy().astype("float32")
  train_x /= train_x.max()

  # one hot encoding
  train_y = data.loc[:, ['class']]
  train_y = np.where(train_y == b'black', 0, 1)
  train_y = to_one_hot(train_y)
  return train_x, train_y


def simple_plot(x, y):
  color = {0: 'b', 1: 'r'}
  y = np.argmax(y, axis=1)
  y = [color[i] for i in y]
  plt.scatter(x[:, 0], x[:, 1], c=y)
  plt.show()


# demo
def binary_classification():

  # prepare train data
  data_dir = "data/examples"
  train_data_path = os.path.join(data_dir, 'training.arff')
  train_x, train_y = load_arff(train_data_path)

  simple_plot(train_x, train_y)

  model = Sequential()
  model.add(Dense(30, activation='relu', input_shape=(2,)))
  model.add(Dense(30, activation='relu'))
  model.add(Dense(2, activation='softmax'))
  model.compile(
    optimizer=Adam(learning_rate=0.1),
    loss='binary_crossentropy',
    metrics=["accuracy"]
  )
  model.fit(train_x, train_y, batch_size=1024, epochs=500)


  # predict
  test_data_path = os.path.join(data_dir, 'test.arff')
  test_x, test_y = load_arff(test_data_path)

  y_hat = model.predict(test_x)
  simple_plot(test_x, y_hat)

import pandas as pd

from demo.classification_data_generator.classification_data_generator import generate_data
from demo.classification_data_generator.generators import HurricaneGenerator, CircleGenerator, CheckerGenerator
from misc import split_data


def multi_classification():
  X, Y = generate_data(3, 5000, HurricaneGenerator)
  (train_x, train_y), (test_x, test_y) = split_data(X, Y, ratio=0.8, random=True)
  train_x = pd.DataFrame(train_x, columns=['x', 'y'])
  train_y = pd.DataFrame(train_y, columns=['class'])
  out = pd.concat((train_x, train_y), axis=1)
  out.to_csv('train_data.csv')

  test_x = pd.DataFrame(test_x, columns=['x', 'y'])
  test_y = pd.DataFrame(test_y, columns=['class'])
  out = pd.concat((test_x, test_y), axis=1)
  out.to_csv('test_data.csv')


if __name__ == '__main__':
  multi_classification()
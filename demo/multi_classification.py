import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from misc.utils import split_data


def multi_classification(csv_file_path):
  """assuming the csv file has columns: x, y, class"""
  df = pd.read_csv(csv_file_path)
  X = df[['x', 'y']].to_numpy()
  Y = df['class'].to_numpy().reshape((-1, 1))

  plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, marker='o')
  plt.show()

  (train_x, train_y), (test_x, test_y) = split_data(X, Y, ratio=0.75, random=True)


multi_classification('../data/examples/multi_classification/test_data.csv')
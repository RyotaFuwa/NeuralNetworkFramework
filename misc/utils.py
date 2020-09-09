import numpy as np


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


def split_data(X, Y=None, ratio=0.8, random=True):
  if Y is not None and X.shape[0] != Y.shape[0]:
    return
  num_of_sample = X.shape[0]
  separate_point = int(num_of_sample * ratio)
  indices = np.arange(num_of_sample)
  if random:
    np.random.shuffle(indices)
  if Y is None:
    return X[indices[:separate_point]], X[indices[separate_point:]]
  return (X[indices[:separate_point]], Y[indices[:separate_point]]), \
         (X[indices[separate_point:]], Y[indices[separate_point:]])
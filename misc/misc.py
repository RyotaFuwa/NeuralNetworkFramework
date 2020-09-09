import time
import numpy as np


def random_like(w: np.ndarray, mue=0.0, sigma=1.0):
  def helper(w: np.ndarray, place: list):
    if w.dtype == 'O':
      new_place = []
      for obj in w:
        helper(obj, new_place)
      place.append(np.array(new_place))
    else:
      value = (np.random.randn(*w.shape) - mue) * sigma
      place.append(value)
  out = []
  helper(w, out)
  return np.array(out[0])


def measure_time(f, *args, **kwargs):
  start = time.time()
  f(*args, **kwargs)
  return time.time() - start


def debug(a: np.ndarray, b: np.ndarray):
  return np.all(a == b)


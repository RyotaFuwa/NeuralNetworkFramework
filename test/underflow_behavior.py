import numpy as np


def print_underflow_value():
  """the value is going to be 0.0 after underflow"""
  const = np.array([0.9, 0.999], dtype='float32')
  i = 0
  while True:
    i += 1
    value = np.power(const, i)
    print(value)


print_underflow_value()

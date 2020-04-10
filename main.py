import numpy as np

from activations import ReLU
from layers import Input, Forward, Dropout


def main():
  i = Input(10)
  x = Forward(10, ReLU)(i)
  x = Dropout(0.2)(x)
  x = Forward(10, ReLU)(x)
  pass


if __name__ == '__main__':
  main()

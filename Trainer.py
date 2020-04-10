from typing import Set, Callable
from optimizers import Optimizer


class Trainer(object):
  CONST: Set[str] = {'loss', 'optimizer'}
  loss: Callable
  optimizer: Optimizer

  batch_size: int = 1
  epochs: int = 1
  shuffle: bool = True
  validation_split: float = 0.0

  def __init__(self, loss: Callable, optimizer: Optimizer, **kwargs):
    self.loss = loss
    self.optimizer = optimizer
    self.set_config(kwargs)

  def __setattr__(self, key, value):
    if key in self.CONST:
      print("Not Allowed To Modify That Value")
    else:
      self.__dict__[key] = value

  def set_config(self, **kwargs):
    for k, v in kwargs:
      self.__setattr__(k, v)


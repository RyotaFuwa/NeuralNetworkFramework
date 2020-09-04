from typing import Set, Callable

from losses import Loss
from optimizers import Optimizer


class Trainer(object):
  _loss: Callable
  _optimizer: Optimizer

  batch_size: int = 1
  epochs: int = 1
  shuffle: bool = True
  validation_split: float = 0.0
  info = True

  def __init__(self, loss: Loss, optimizer: Optimizer, **kwargs):
    self.loss = loss
    self.optimizer = optimizer
    self.set_config(**kwargs)

  def set_config(self, **kwargs):
    for k, v in kwargs.items():
      self.__dict__[k] = v




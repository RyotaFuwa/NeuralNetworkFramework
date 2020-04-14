from typing import Set, Callable
from optimizers import Optimizer


class Trainer(object):
  _loss: Callable
  _optimizer: Optimizer

  batch_size: int = 1
  epochs: int = 1
  shuffle: bool = True
  validation_split: float = 0.0
  info = True

  def __init__(self, loss: Callable, optimizer: Optimizer, **kwargs):
    self._loss = loss
    self._optimizer = optimizer
    self.set_config(**kwargs)

  @property
  def loss(self):
    pass
  @loss.getter
  def loss(self):
    return self._loss

  @property
  def optimizer(self):
    pass
  @optimizer.getter
  def optimizer(self):
    return self._optimizer

  def set_config(self, **kwargs):
    for k, v in kwargs.items():
      self.__dict__[k] = v



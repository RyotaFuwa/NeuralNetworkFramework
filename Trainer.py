from typing import Union, Type

from losses import Loss, loss_load
from optimizers import Optimizer, optimizer_load


class Trainer(object):
  loss: Type[Loss]
  optimizer: Optimizer

  batch_size: int = 1
  epochs: int = 1
  shuffle: bool = True
  validation_split: float = 0.0
  info = True

  def __init__(self, loss: str, optimizer: Union[str, Optimizer], metrics: list = [], **kwargs):
    self.loss = loss_load(loss)
    if self.loss is None:
      raise Exception('not valid loss function')

    if isinstance(optimizer, str):
      self.optimizer = optimizer_load(optimizer)
      if optimizer is None:
        raise Exception('not valid optimizer function')
    else:
      self.optimizer = optimizer

    self.metrics = ['loss'] + metrics
    self.set_config(**kwargs)

  def set_config(self, **kwargs):
    for k, v in kwargs.items():
      self.__dict__[k] = v

    # condition
    if self.batch_size == 0:
      raise Exception("batch_size can not be 0")
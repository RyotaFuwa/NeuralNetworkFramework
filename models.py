from abc import ABC, abstractmethod
from typing import Callable, List, Set, Tuple, Union
import numpy as np
from Error import ShapeIncompatible, InputNotMatch
from Trainer import Trainer
from layers import Input, Layer, Forward
from misc import TYPE_INPUT, TYPE_LAYER
from queue import Queue

"""Design Assumption
the interface is inspired by keras in tensorflow package, and calculation are mainly
implemented by the use of numpy.
"""

# TODO aggregate layers
# TODO dw
# TODO optimizer's update


class Model(ABC):
  _I: TYPE_INPUT
  _O: TYPE_LAYER

  def __init__(self, ins: TYPE_INPUT, outs: TYPE_LAYER):
    if type(ins) != tuple:
      self._I = (ins,)
    else:
      self._I = ins
    if type(outs) != tuple:
      self._O = (outs,)
    else:
      self._O = outs

    self._L = []
    queue = Queue()
    for l in outs:
      queue.put(l)
    while queue.not_empty:
      current_layer = queue.get()
      self._L.append(current_layer)
      if current_layer != (None,):
        queue.put(current_layer.previous)

  def learn(self, x: np.ndarray, y: np.ndarray, trainer: Trainer):
    trainer.optimizer.initialize(self._L)

    num_of_samples = x.shape[0]
    num_of_cycles = num_of_samples * trainer.epochs // trainer.batch_size
    if num_of_cycles < 1:
      num_of_cycles = 1

    for _ in range(num_of_cycles):
      _in = self._get_Batch(x, batch_size=trainer.batch_size, shuffle=trainer.shuffle)
      _out = self.f(_in)
      cost = Trainer.loss(_out, y)
      trainer.optimizer(_in, cost)

  @abstractmethod
  def predict(self, x: np.ndarray):
    """calculate output"""

  def __call__(self, x: np.ndarray):
    """alias for predict"""
    return self.predict(x)

  @abstractmethod
  def f(self, x: np.ndarray, predict=False):
    """calculate"""

  @staticmethod
  def _get_Batch(x: np.ndarray, batch_size: int, shuffle: bool = False) -> np.ndarray:
    if shuffle:
      index = np.random.randint(0, x.shape[0], size=(batch_size,))
    else:
      index = np.arange(batch_size)
    return x[index, :]


class ANN(Model):
  def predict(self, x: np.ndarray):
    self.f(x, predict=True)

  def f(self, x: np.ndarray, predict=False):
    for l in self._L:
      if predict and l.ONLY_IN_TRAINING:
        continue
      x = l.f(x)
    return x

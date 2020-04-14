from abc import ABC, abstractmethod
from typing import Callable, List, Set, Tuple, Union
import numpy as np
from _layers import Layer, Input
from Error import ShapeIncompatible, InputNotMatch
from Trainer import Trainer
from queue import Queue

TYPE_LAYER = Union[Layer, Tuple[Layer]]
TYPE_INPUT = Union[Input, Tuple[Input]]

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
  _layers: Tuple[Layer]  # sequence of layers

  trainer: Trainer
  cycle: int

  def __init__(self, ins: TYPE_INPUT, outs: TYPE_LAYER):
    if type(ins) != tuple:
      self._I = (ins,)
    else:
      self._I = ins
    if type(outs) != tuple:
      self._O = (outs,)
    else:
      self._O = outs

    self.cycle = 0

    rev = []
    queue = Queue()
    for l in self._O:
      queue.put(l)
    while not queue.empty():
      current_layer = queue.get()
      rev.append(current_layer)
      for l in current_layer.prev:
        queue.put(l)

    rev.reverse()
    self._layers = tuple(rev)

  def train(self, x: np.ndarray, y: np.ndarray, trainer: Trainer):
    self.trainer = trainer
    trainer.optimizer.initialize(self._layers)

    num_of_samples = x.shape[0]
    num_of_cycles = num_of_samples * trainer.epochs // trainer.batch_size
    if num_of_cycles < 1:
      num_of_cycles = 1

    for i in range(num_of_cycles):
      self.cycle = i
      epoch = int(i * trainer.batch_size / num_of_samples)
      batch_in, label = self.get_Batch(x, y, batch_size=trainer.batch_size, shuffle=trainer.shuffle)
      out = self.f(batch_in)
      loss_value = trainer.loss.f(out, label)
      dy = trainer.loss.df()  # dy: du/d(out) dx.shape == label.shape
      self.df(dy)
      trainer.optimizer.update()
      if trainer.info:
        print("epoch: {}:: Loss: {:<10.3f}".format(
          epoch, loss_value))

  def __call__(self, x: np.ndarray):
    """alias for predict"""
    return self.predict(x)

  def predict(self, x: np.ndarray):
    self.f(x, predict=True)

  @abstractmethod
  def f(self, ins: np.ndarray, predict=False):
    """forward propagation"""

  @abstractmethod
  def df(self, l):
    """back propagation"""

  def get_Batch(self, x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    if shuffle:
      index = np.random.randint(0, x.shape[0], size=(batch_size,))
    else:
      index = np.arange(batch_size)
      x = np.roll(x, batch_size * self.cycle)
    return x[index, :], y[index, :]


class FNN(Model):

  def f(self, x: np.ndarray, predict=False):
    for l in self._layers:
      if predict and l.ONLY_IN_TRAINING:
        continue
      x = l.f(x)
    return x

  def df(self, dy):
    rev_layers = self._layers[::-1]
    for l in rev_layers:
      dy = l.df(dy)


class ComplicatedModel(Model):  # TODO  Build Not Sequential Model
  def f(self, x: np.ndarray, predict=False):
    pass
  def df(self, loss_value):
    pass


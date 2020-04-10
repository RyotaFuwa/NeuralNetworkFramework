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

  @abstractmethod
  def learn(self, x: np.ndarray, y: np.ndarray, trainer: Trainer):
    """learning process"""

  @abstractmethod
  def predict(self, x: np.ndarray):
    """calculate output"""

  def summary(self):
    title_txt = self.__class__.__name__ + "::\n"
    header_txt = "--->\n"
    layer_txt = "\n".join(["{} Layer({})| ".format(i, "L" if layer.LEARNABLE else "NL").ljust(11) + str(layer)
                           for i, layer in enumerate(self._layers)])
    footer_txt = "<---\n"
    return title_txt + header_txt + layer_txt + footer_txt

  def __call__(self, x: np.ndarray):
    """alias for predict"""
    return self.predict(x)

  def __repr__(self):
    """alias for summary"""
    return self.summary()

  @staticmethod
  def _get_Batch(x: np.ndarray, batch_size: int, shuffle: bool = False) -> np.ndarray:
    if shuffle:
      index = np.random.randint(0, x.shape[0], size=(batch_size,))
    else:
      index = np.arange(batch_size)
    return x[index, :]


class ANN(Model):

  def learn(self, x: np.ndarray, y: np.ndarray, trainer: Trainer):
    trainer.optimizer.initialize(self._I, self._O)

    num_of_samples = x.shape[0]
    num_of_cycles = num_of_samples * trainer.epochs // trainer.batch_size
    if num_of_cycles < 1:
      num_of_cycles = 1
    for _ in range(num_of_cycles):
      _in = self._get_Batch(x, batch_size=trainer.batch_size, shuffle=trainer.shuffle)
      _out = self._calculate(_in)
      cost = Trainer.loss(_out, y)

      # TODO
      self.trainer.optimizer(_in, cost)

  def predict(self, x: np.ndarray):
    self._calculate(x, predict=True)

  def _calculate(self, x: np.ndarray, predict=False):
    for layer in self._layers: #TODO
      if predict and layer.ONLY_IN_TRAINING:
        continue
      x = layer.calculate(x)
    return x

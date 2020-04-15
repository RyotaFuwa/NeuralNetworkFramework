from abc import ABC, abstractmethod
from typing import Callable, List, Set, Tuple, Union
import numpy as np
from _layers import Layer, SequenceLayer, Input
from Error import ShapeIncompatible, InputNotMatch
from Trainer import Trainer
from losses import CrossEntropy, SoftmaxWithCrossEntropy
from activations import Softmax
from queue import Queue
from misc import debug

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
  _L: Tuple[Layer]  # sequence of layers

  trainer: Trainer
  cycle: int

  @abstractmethod
  def __init__(self, i: SequenceLayer, o: SequenceLayer, trainer):
    self._I = i
    self._O = o
    self.trainer = trainer

  def train(self, x: np.ndarray, y: np.ndarray):
    self.trainer.optimizer.initialize(self._L)

    # setup variables
    num_of_samples = x.shape[0]
    num_of_cycles = num_of_samples * self.trainer.epochs // self.trainer.batch_size
    if num_of_cycles < 1:
      num_of_cycles = 1

    # main loop
    for i in range(num_of_cycles):
      self.cycle = i
      epoch = int(i * self.trainer.batch_size / num_of_samples)

      # get mini-batch
      batch_in, label = self.get_Batch(x, y, batch_size=self.trainer.batch_size, shuffle=self.trainer.shuffle)

      # forward calculation
      out = self.f(batch_in)

      # calculate loss values
      loss_value = self.trainer.loss.f(out, label)

      # back propagation
      dy = self.trainer.loss.df()  # dy: du/d(out) dx.shape == label.shape
      self.df(dy)

      # update parameters
      self.trainer.optimizer.update()

      # print info
      if self.trainer.info:
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


class SequenceModel(Model, ABC):
  def __init__(self, i: SequenceLayer, o: SequenceLayer, trainer: Trainer):
    super().__init__(i, o, trainer)
    # efficiency optimization with softmax and cross entropy.
    if type(trainer.loss) == CrossEntropy and type(self._O) == Softmax:
      self._O = self._O.prev
      self.trainer.loss = SoftmaxWithCrossEntropy()

    current_layer = i
    L = [i]
    while current_layer is not self._O:
      current_layer = current_layer.next
      L.append(current_layer)
    self._L = tuple(L)

    self.cycle = 0


class FNN(SequenceModel):

  def f(self, x: np.ndarray, predict=False):
    for l in self._L:
      if predict and l.ONLY_IN_TRAINING:
        continue
      x = l.f(x)
    return x

  def df(self, dy):
    rev_L = self._L[::-1]
    for l in rev_L:
      dy = l.df(dy)


class AggregateLayer(Model, ABC):  # TODO  Build Not Sequential Model
  def __init__(self, ins: TYPE_INPUT, outs: TYPE_LAYER):
    if type(ins) != tuple:
      self._I = (ins,)
    else:
      self._I = ins
    if type(outs) != tuple:
      self._O = (outs,)
    else:
      self._O = outs

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

    self.cycle = 0

  def f(self, x: np.ndarray, predict=False):
    pass
  def df(self, loss_value):
    pass


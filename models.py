from abc import ABC, abstractmethod
from typing import Tuple, Union
import numpy as np
from layers import Layer, SequenceLayer, Input
from Trainer import Trainer
from losses import CrossEntropy, SoftmaxWithCrossEntropy
from activations import Softmax

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
  def __init__(self, i: SequenceLayer, o: SequenceLayer, trainer: Trainer):
    self._I = i
    self._O = o
    self.trainer = trainer

    current = self._I
    while current is not None:
      if current.LEARNABLE:
        current.updater = trainer.optimizer(current)
      current = current.next

  def train(self, x: np.ndarray, y: np.ndarray):

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
      batch_in, label = self.get_batch(x, y, batch_size=self.trainer.batch_size, shuffle=self.trainer.shuffle)

      # forward calculation
      out = self.f(batch_in)

      # print info
      if self.trainer.info:
        # calculate loss values
        loss_value = self.trainer.loss.f(out, label)
        print("epoch: {}:: Loss: {:<10.3f}".format(
          epoch, loss_value))

      # back propagation
      dy = self.trainer.loss.df()  # dy: du/d(out) dx.shape == label.shape
      self.df(dy)

    print(self._O.w)

  def __call__(self, x: np.ndarray):
    """alias for predict"""
    return self.predict(x)

  def predict(self, x: np.ndarray):
    return self.f(x, predict=True)

  @abstractmethod
  def f(self, ins: np.ndarray, predict=False):
    """forward propagation"""

  @abstractmethod
  def df(self, dl):
    """back propagation: calculate dy and update layer.w"""

  def get_batch(self, x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> Tuple[
    np.ndarray, np.ndarray]:
    if shuffle:
      index = np.random.randint(0, x.shape[0], size=(batch_size,))
    else:
      index = np.arange(batch_size)
      x = np.roll(x, batch_size * self.cycle)
    return x[index, :], y[index, :]


class SequenceModel(Model, ABC):
  def __init__(self, i: SequenceLayer, o: SequenceLayer, trainer: Trainer):
    super().__init__(i, o, trainer)
    self.cycle = 0

    # efficiency optimization with softmax and cross entropy.
    if type(trainer.loss) == CrossEntropy and type(self._O) == Softmax:
      self._O = self._O.prev
      self.trainer.loss = SoftmaxWithCrossEntropy()


class FNN(SequenceModel):
  def f(self, x: np.ndarray, predict=False):
    current_layer = self._I
    while current_layer is not None:
      if not (predict and current_layer.ONLY_IN_TRAINING):
        x = current_layer.f(x)
      current_layer = current_layer.next
    return x

  def df(self, dy):
    current_layer = self._O
    while current_layer is not None:
      dy = current_layer.df(dy)
      current_layer = current_layer.prev
    return dy



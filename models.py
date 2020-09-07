from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional
import numpy as np
from layers import Layer, SequentialLayer, Input
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
  history: dict

  @abstractmethod
  def __init__(self, i: SequentialLayer, o: SequentialLayer, trainer: Trainer):
    self._I = i
    self._O = o
    self.trainer = trainer
    self.str = ''
    self.history: {'loss': []}

    current = self._I
    while current is not None:
      self.str += current.__str__() + '\n'
      if current.LEARNABLE:
        current.updater = trainer.optimizer(current)
      current = current.next

  def train(self, x: np.ndarray, y: np.ndarray):
    self.history = {'loss': []}

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
      loss_value = self.trainer.loss.f(out, label)
      self.history['loss'].append(loss_value)

      # print info
      if self.trainer.info:
        # calculate loss values
        print("epoch: {}:: Loss: {:<10.3f}".format(
          epoch, loss_value))

      # back propagation
      dy = self.trainer.loss.df()  # dy: du/d(out) dx.shape == label.shape
      self.df(dy)

  def __call__(self, x: np.ndarray):
    return self.predict(x)

  def __str__(self):
    return self.summary()

  def predict(self, x: np.ndarray):
    return self.f(x, predict=True)

  def evaluate(self, y_hat: np.ndarray, test_y: np.ndarray):
    """just return the accuracy of the predict"""
    compared = y_hat == test_y
    return np.count_nonzero(compared) / compared.shape[0]

  def summary(self):
    return self.str


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


class Sequential(Model):
  softmax_and_crossentropy: bool
  softmax: Optional[Softmax]

  def __init__(self, i: SequentialLayer, o: SequentialLayer, trainer: Trainer):
    super().__init__(i, o, trainer)
    self.cycle = 0
    self.softmax_and_crossentropy = False

    # efficiency optimization with softmax and cross entropy. TODO: make sure softmax is called in predict
    if type(trainer.loss) == CrossEntropy and type(self._O) == Softmax:
      self.softmax_and_crossentropy = True
      self.sofmax = self._O
      self._O = self._O.prev
      self._O._next._prev = None
      self._O._next = None
      self.trainer.loss = SoftmaxWithCrossEntropy()

  def f(self, x: np.ndarray, predict=False):
    current_layer = self._I
    while current_layer is not None:
      if not (predict and current_layer.ONLY_IN_TRAINING):
        x = current_layer.f(x)
      current_layer = current_layer.next
    if self.softmax_and_crossentropy and predict:
      x = self.sofmax.f(x)
    return x

  def df(self, dy):
    current_layer = self._O
    while current_layer is not None:
      dy = current_layer.df(dy)
      current_layer = current_layer.prev
    return dy


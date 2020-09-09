from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional
import numpy as np
from layers import _Layer, Input, Layer
from Trainer import Trainer
from losses import CrossEntropy, SoftmaxWithCrossEntropy, SparseCrossEntropy
from activations import Softmax

"""Design Assumption
the interface is inspired by keras in tensorflow package, and calculation are mainly
implemented by the use of numpy.
"""

METRICS = ['accuracy']


class Model(ABC):
  head: _Layer
  tail: _Layer
  trainer: Trainer

  loss_value: np.float

  current_epoch: int
  history: dict

  @abstractmethod
  def __init__(self, i: _Layer, o: _Layer, trainer: Trainer):
    if not isinstance(i, Input):
      raise Input
    self.head = i
    self.tail = o
    self.trainer = trainer
    self.history = {}
    self.initialize()

  @abstractmethod
  def f(self, ins: np.ndarray, training=True):
    """forward propagation"""

  @abstractmethod
  def df(self, y_hat: np.ndarray, y: np.ndarray):
    """back propagation: calculate dy and update layer.w"""

  def initialize(self):
    self.current_epoch = 0

    # initialize history
    for key in self.trainer.metrics:
      self.history[key] = []

    # initialize weights for layers
    current = self.head
    while current is not None:
      if isinstance(current, Layer):
        current.initialize()
        current.updater = self.trainer.optimizer(current)
      current = current.next

  def fit(self, x: np.ndarray, y: np.ndarray, reset=False):
    if reset:
      self.initialize()

    # setup variables
    num_of_samples = x.shape[0]
    num_of_cycles = num_of_samples * self.trainer.epochs // self.trainer.batch_size
    if num_of_cycles < 1:
      num_of_cycles = 1

    # main loop
    for i in range(num_of_cycles):
      new_epoch = int(i * self.trainer.batch_size / num_of_samples)
      epoch_changed = self.current_epoch != new_epoch
      if epoch_changed:
        self.current_epoch = new_epoch

      x_min_batch, y_min_batch = self.get_batch(x, y, batch_size=self.trainer.batch_size, shuffle=self.trainer.shuffle, step=i)

      # forward propagation (calculation)
      y_hat_min_batch = self.f(x_min_batch)

      # calculate loss value
      loss_value = self.trainer.loss.f(y_hat_min_batch, y_min_batch)

      # back propagation (optimization)
      self.df(y_hat_min_batch, y_min_batch)

      # data handling and hold metric in history
      if epoch_changed:
        if 'loss' in self.history:
          self.history['loss'].append(loss_value)

        if 'accuracy' in self.history:
          if self.trainer.loss is CrossEntropy:
            y_hat_predict = np.argmax(y_hat_min_batch, axis=-1)
            y_predict = np.argmax(y_min_batch, axis=-1)
          elif self.trainer.loss is SparseCrossEntropy:
            y_hat_predict = np.argmax(y_hat_min_batch, axis=-1)
            y_predict = y_min_batch.reshape((-1))
          else:
            y_hat_predict = y_hat_min_batch
            y_predict = y_min_batch
          accuracy = np.count_nonzero(y_hat_predict == y_predict) / y_hat_predict.shape[0]
          if self.current_epoch > 1:
            accuracy = (self.history['accuracy'][-1] * (self.current_epoch - 1) + accuracy) / self.current_epoch
          self.history['accuracy'].append(accuracy)

        if self.trainer.info:
          # calculate loss values
          info_line = f"epoch: {self.current_epoch}:: "
          for key in self.trainer.metrics:
            info_line += f" - {key}: {self.history[key][-1]:8.4f}"
          print(info_line)

  def predict(self, x: np.ndarray):
    return self.f(x, training=False)

  def summary(self, stdout=True):
    out = ''
    current = self.head
    while current is not None:
      out += current.__str__() + '\n'
      current = current.next
    if stdout:
      print(out)
    return out

  @staticmethod
  def get_batch(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True, step: int = 0) -> Tuple[
    np.ndarray, np.ndarray]:
    if shuffle:
      index = np.random.randint(0, x.shape[0], size=(batch_size,))
    else:
      index = np.arange(batch_size)
      x = np.roll(x, batch_size * step)
    return x[index], y[index]


class Sequential(Model):
  softmax_with_crossentropy: bool

  def __init__(self, i: _Layer, o: _Layer, trainer: Trainer):
    super().__init__(i, o, trainer)
    self.cycle = 0
    self.softmax_with_crossentropy = (trainer.loss is CrossEntropy) and isinstance(self.tail, Softmax)

  def f(self, x: np.ndarray, training=True):
      current_layer = self.head
      while current_layer is not None:
        x = current_layer.f(x, training)
        current_layer = current_layer.next
      return x

  def df(self, y_hat: np.ndarray, y: np.ndarray):
    if self.softmax_with_crossentropy:
      dl = SoftmaxWithCrossEntropy.df(y_hat, y)
      current_layer = self.tail.prev
    else:
      dl = self.trainer.loss.df(y_hat, y)
      current_layer = self.tail

    while current_layer is not None:
      dl = current_layer.df(dl)
      current_layer = current_layer.prev
    return dl


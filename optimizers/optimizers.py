from .updaters import *
from .learaning_schedules import *
from misc.types import LEARNING_RATE, learning_rate2schedule

"""Optimizer Design
an instance of this class can store the reference to the weights of all layers in the model
as well as other states such as dW depending on the type of optimizer
"""


class Optimizer(ABC):
  step: int
  schedule: Schedule
  clipnorm: float
  clipvalue: float

  @abstractmethod
  def __init__(self, **kwargs):
    self.step = 0
    self.schedule = DefaultSchedule()
    self.clipnorm = kwargs.get('clipnorm')
    self.clipvalue = kwargs.get('clipvalue')

  @abstractmethod
  def __call__(self, layer: Layer):
    """return instance of Updater class defined in Optimizer class"""

  @abstractmethod
  def update(self):
    self.step += 1


class SGD(Optimizer):
  learning_rate: float
  momentum: float

  def __init__(self, learning_rate: LEARNING_RATE = 0.001, momentum: float = 0.0, **kwargs):
    super().__init__(**kwargs)
    self.schedule = learning_rate2schedule(learning_rate)
    self.learning_rate = self.schedule.get_learning_rate(self.step)
    self.momentum = momentum

  def __call__(self, layer: Layer):
    return SGDUpdater(self, layer)

  def update(self):
    super().update()
    self.learning_rate = self.schedule.get_learning_rate(self.step)


class Adam(Optimizer):
  def __init__(self, learning_rate: LEARNING_RATE = 0.001, beta1: float = 0.9, beta2: float = 0.999, **kwargs):
    super().__init__(**kwargs)
    self.schedule = learning_rate2schedule(learning_rate)
    self.learning_rate = self.schedule.get_learning_rate(self.step)
    self.beta1 = beta1
    self.beta2 = beta2

  def __call__(self, layer: Layer):
    return AdamUpdater(self, layer)

  def update(self):
    super().update()
    self.learning_rate = self.schedule.get_learning_rate(self.step)


REGISTERED_OPTIMIZERS = {
  'sgd': SGD,
  'adam': Adam,
}


def optimizer_load(key: str):
  if key in REGISTERED_OPTIMIZERS:
    return REGISTERED_OPTIMIZERS[key]()
  return None

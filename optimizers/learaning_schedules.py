from abc import ABC, abstractmethod


class Schedule(ABC):
  @abstractmethod
  def __init__(self):
    """initialize params needed for the schedule"""

  @abstractmethod
  def get_learning_rate(self, step: int) -> float:
    """return updated learning_rate"""


class DefaultSchedule(Schedule):
  def __init__(self):
    super().__init__()

  def get_learning_rate(self, step: int) -> float:
    pass


class NoDecay(Schedule):
  def __init__(self, initial_learning_rate: float = 0.001):
    super().__init__()
    self.initial_learning_rate = initial_learning_rate

  def get_learning_rate(self, step: int) -> float:
    return self.initial_learning_rate


class ExponentialDecay(Schedule):
  def __init__(self, initial_learning_rate: float = 0.001,  decay_steps: int = 100000, decay_rate: float = 0.96):
    super().__init__()
    self.initial_learning_rate = initial_learning_rate
    self.decay_steps = decay_steps
    self.decay_rate = decay_rate

  def get_learning_rate(self, step: int) -> float:
    return self.initial_learning_rate * self.decay_rate ** (step / self.decay_steps)



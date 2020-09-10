from typing import Union, Tuple
from optimizers.learaning_schedules import Schedule, NoDecay

SHAPE = Union[int, Tuple[int]]
LEARNING_RATE = Union[float, Schedule]


def shape2tuple(shape: SHAPE):
  if type(shape) == int:
      shape = (shape,)
  return shape


def learning_rate2schedule(learning_rate: LEARNING_RATE):
  schedule = learning_rate
  if not isinstance(learning_rate, Schedule):
    schedule = NoDecay(learning_rate)
  return schedule



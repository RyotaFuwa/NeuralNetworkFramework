from typing import Union, Tuple

SHAPE = Union[int, Tuple[int]]


def shape2tuple(shape: SHAPE):
  if type(shape) == int:
      shape = (shape,)
  return shape



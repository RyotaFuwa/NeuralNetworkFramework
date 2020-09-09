
class ShapeIncompatible(Exception):
  def __init__(self, msg=''):
    self.msg = msg


class NetworkNotReady(Exception):
  def __init__(self, msg=''):
    self.msg = msg


class InputNotMatch(Exception):
  pass



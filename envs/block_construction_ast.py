import numpy as np
import numpy.random as npr

import objects

class Cube(object):
  def eval(self, obj):
    return obj == object_types.CUBE

class Above(object):
  def __init__(self, left, right):
    self.left = left
    self.right = right

  def eval(self, x, y):
    return self.left.eval(x) and \
           self.right.eval(y) and \
           np.allclose(x.getPos[:2], y.getPos[:2])

class Any(object):
  def eval(self):
    pass

class All(object):
  def eval(self, ):
    return x.eval() and y.eval()

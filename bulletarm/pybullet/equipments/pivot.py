import pybullet as pb
import numpy as np

class Pivot(object):
  '''

  '''
  def __init__(self):
    self.id = None

  def initialize(self, pos=(0,0,0), rot=(0,0,0,1), size=(0.3, 0.3, 0.1)):
    ''''''

  def reset(self, pos=(0,0,0), rot=(0,0,0,1)):
    ''''''

  def getPose(self):
    ''''''

  def remove(self):
    ''''''
    if self.id:
      pb.removeBody(self.id)
    self.id = None

import pybullet as pb
import numpy as np
import scipy
import numpy.random as npr
from copy import deepcopy
from bulletarm.pybullet.utils import pybullet_util
from bulletarm.pybullet.utils.constants import NoValidPositionException

from bulletarm.envs.deconstruct_envs.deconstruct_env import DeconstructEnv
from bulletarm.envs.ramp_envs.ramp_base_env import RampBaseEnv
import bulletarm.pybullet.utils.object_generation as pb_obj_generation
from bulletarm.pybullet.utils import constants

class RampDeconstructEnv(DeconstructEnv, RampBaseEnv):
  def __init__(self, config):
    super(RampDeconstructEnv, self).__init__(config)
    self.pick_offset = 0.0
    self.place_offset = 0.015

  def resetRampDeconstructEnv(self):
    self.resetPybulletWorkspace()
    self.resetRamp()
    self.structure_objs = []
    self.generateStructure()
    while not self.checkStructure():
      self.resetPybulletWorkspace()
      self.resetRamp()
      self.structure_objs = []
      self.generateStructure()

  def reset(self):
    self.resetRampDeconstructEnv()
    return self._getObservation()

  def get1BaseXY(self, padding):
    while True:
      pos = self._getValidPositions(padding, 0, [], 1)[0]
      if self.isPosOffRamp(pos):
        break
    return pos

  def get2BaseXY(self, padding, min_dist, max_dist):
    while True:
      pos1 = self._getValidPositions(padding, 0, [], 1)[0]
      if self.isPosOffRamp(pos1):
        break
    if self.random_orientation:
      sample_range = [[pos1[0] - max_dist, pos1[0] + max_dist],
                      [pos1[1] - max_dist, pos1[1] + max_dist]]
    else:
      sample_range = [[pos1[0] - 0.005, pos1[0] + 0.005],
                      [pos1[1] - max_dist, pos1[1] + max_dist]]
    for i in range(100):
      try:
        pos2 = self._getValidPositions(padding, min_dist, [pos1], 1, sample_range=sample_range)[0]
      except NoValidPositionException:
        continue
      dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
      if min_dist < dist < max_dist and self.isPosOffRamp(pos2):
        break
    return pos1, pos2

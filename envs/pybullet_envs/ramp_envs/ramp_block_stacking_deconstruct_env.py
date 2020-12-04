import numpy as np
from itertools import combinations
from envs.pybullet_envs.ramp_envs.ramp_deconstruct_env import RampDeconstructEnv
from helping_hands_rl_envs.simulators import constants

class RampBlockStackingDeconstructEnv(RampDeconstructEnv):
  ''''''
  def __init__(self, config):
    super().__init__(config)

  def generateStructure(self):
    padding = self.max_block_size * 1.5
    while True:
      pos = self._getValidPositions(padding, 0, [], 1)[0]
      if self.isPosOffRamp(pos):
        break
    rot = self._getValidOrientation(self.random_orientation)
    for i in range(self.num_obj):
      self.generateStructureShape((pos[0], pos[1], i * self.max_block_size + self.max_block_size / 2), rot,
                                  constants.CUBE)
    self.wait(50)

  def checkStructure(self):
    ''''''
    return self._checkStack()

  def _checkTermination(self):
    if self.current_episode_steps < (self.num_obj-1)*2:
      return False
    obj_combs = combinations(self.objects, 2)
    for (obj1, obj2) in obj_combs:
      dist = np.linalg.norm(np.array(obj1.getXYPosition()) - np.array(obj2.getXYPosition()))
      if dist < 2.4*self.min_block_size:
        return False
    return True


def createRampBlockStackingDeconstructEnv(config):
  return RampBlockStackingDeconstructEnv(config)
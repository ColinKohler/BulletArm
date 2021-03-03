import os
import pybullet as pb
import copy
import numpy as np
import numpy.random as npr

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.equipments.shelf import Shelf
from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.utils import transformations
from helping_hands_rl_envs.simulators.constants import NoValidPositionException
from helping_hands_rl_envs.planners.shelf_bowl_stacking_planner import ShelfBowlStackingPlanner

class ShelfBowlStackingEnv(PyBulletEnv):
  def __init__(self, config):
    super().__init__(config)
    self.place_offset = 0.04
    self.shelf = Shelf()
    self.object_init_space = np.asarray([[0.3, 0.7],
                                         [-0.4, 0],
                                         [0, 0.40]])
  
  def initialize(self):
    super().initialize()
    self.shelf.initialize(pos=[0.6, 0.3, 0])

    pass

  def reset(self):
    ''''''
    while True:
      self.resetPybulletEnv()
      try:
        self._generateShapes(constants.BOWL, self.num_obj, random_orientation=self.random_orientation)
      except NoValidPositionException as e:
        continue
      else:
        break
    return self._getObservation()

  def anyObjectOnTarget1(self):
    for obj in self.objects:
      if self.shelf.isObjectOnTarget1(obj):
        return True
    return False

  def _checkTermination(self):
    return self._checkStack() and self.anyObjectOnTarget1()

  def _getValidOrientation(self, random_orientation):
    if random_orientation:
      orientation = pb.getQuaternionFromEuler([0., 0., np.pi * -(np.random.random_sample() * 0.5 + 0.5)])
    else:
      orientation = pb.getQuaternionFromEuler([0., 0., 0.])
    return orientation
    
  def getValidSpace(self):
    return self.object_init_space

def createShelfBowlStackingEnv(config):
  return ShelfBowlStackingEnv(config)
  

if __name__ == '__main__':
  object_init_space = np.asarray([[0.3, 0.7],
                          [-0.4, 0.4],
                          [0, 0.40]])
  env_config = {'object_init_space': object_init_space, 'max_steps': 10, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyzrrr', 'num_objects': 5, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'object_init_space_check': 'point', 'physics_mode': 'slow'}
  planner_config = {'random_orientation': True}

  env = ShelfBowlStackingEnv(env_config)
  planner = ShelfBowlStackingPlanner(env, planner_config)
  s, in_hand, obs = env.reset()
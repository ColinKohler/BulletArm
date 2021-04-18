import os
import pybullet as pb
import copy
import numpy as np
import numpy.random as npr

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.equipments.box import Box
from helping_hands_rl_envs.simulators.pybullet.objects.plate import PLACE_RY_OFFSET, PLACE_Z_OFFSET
from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.utils import transformations
from helping_hands_rl_envs.simulators.constants import NoValidPositionException
from helping_hands_rl_envs.planners.bottle_tray_planner import BottleTrayPlanner

class BottleTrayEnv(PyBulletEnv):
  def __init__(self, config):
    super().__init__(config)
    self.place_offset = 0.12

    self.box = Box()
    self.object_init_space = np.asarray([[0.3, 0.7],
                                         [-0.3, 0.1],
                                         [0, 0.40]])
    self.box_pos = [0.5, 0.2, 0]
    self.box_size = [0.2, 0.15, 0.05]
    self.box_range = np.array([[self.box_pos[0]-self.box_size[0]/2, self.box_pos[0]+self.box_size[0]/2],
                               [self.box_pos[1]-self.box_size[1]/2, self.box_pos[1]+self.box_size[1]/2]])
    self.x_candidate = np.linspace(self.box_range[0][0], self.box_range[0][1], 7)[1:-1:2]
    self.y_candidate = np.linspace(self.box_range[1][0], self.box_range[1][1], 5)[1:-1:2]
    self.place_pos_candidate = [
      [self.x_candidate[0], self.y_candidate[0]], [self.x_candidate[0], self.y_candidate[1]],
      [self.x_candidate[1], self.y_candidate[0]], [self.x_candidate[1], self.y_candidate[1]],
      [self.x_candidate[2], self.y_candidate[0]], [self.x_candidate[2], self.y_candidate[1]],
    ]
    pass

  def getValidSpace(self):
    return self.object_init_space

  def initialize(self):
    super().initialize()
    self.box.initialize(pos=self.box_pos, size=self.box_size)


  def reset(self):
    while True:
      self.resetPybulletEnv()
      try:
        self._generateShapes(constants.BOTTLE, self.num_obj, random_orientation=self.random_orientation, padding=self.min_boarder_padding, min_distance=self.min_object_distance)
      except NoValidPositionException:
        continue
      else:
        break
    return self._getObservation()

  def _checkTermination(self):
    for obj in self.objects:
      if self.isObjInBox(obj):
        continue
      else:
        return False
    return True

  def isObjInBox(self, obj):
    return self.box_range[0][0] < obj.getPosition()[0] < self.box_range[0][1] and self.box_range[1][0] < obj.getPosition()[1] < self.box_range[1][1]

  def getObjsOutsideBox(self):
    return list(filter(lambda obj: not self.isObjInBox(obj), self.objects))

def createBlockBinPackingEnv(config):
  return BottleTrayEnv(config)

if __name__ == '__main__':
  workspace = np.asarray([[0.2, 0.8],
                          [-0.3, 0.3],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 2, 'action_sequence': 'pxyr', 'num_objects': 6, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'object_init_space_check': 'point', 'physics_mode': 'fast', 'object_scale_range': (0.6, 0.6)}
  planner_config = {'random_orientation': True}

  env = BottleTrayEnv(env_config)
  planner = BottleTrayPlanner(env, planner_config)
  s, in_hand, obs = env.reset()
  while True:
    action = planner.getNextAction()
    obs, reward, done = env.step(action)
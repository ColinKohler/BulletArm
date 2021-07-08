import os
import pybullet as pb
import copy
import numpy as np
import numpy.random as npr

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.equipments.container_box import ContainerBox
from helping_hands_rl_envs.simulators.pybullet.objects.plate import PLACE_RY_OFFSET, PLACE_Z_OFFSET
from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.utils import transformations
from helping_hands_rl_envs.simulators.constants import NoValidPositionException
from helping_hands_rl_envs.planners.block_bin_packing_planner import BlockBinPackingPlanner

class BlockBinPackingEnv(PyBulletEnv):
  def __init__(self, config):
    super().__init__(config)
    self.box = ContainerBox()
    self.box_rz = 0
    self.box_pos = [0.60, 0.12, 0]
    self.box_size = [0.23*self.block_scale_range[1], 0.15*self.block_scale_range[1], 0.1]
    self.box_placeholder_pos = []
    self.z_threshold = self.box_size[-1]

  def resetBox(self):
    self.box_rz = np.random.random_sample() * np.pi
    self.box_pos = self._getValidPositions(np.linalg.norm([self.box_size[0], self.box_size[1]]), 0, [], 1)[
      0]
    self.box_pos.append(0)
    self.box.reset(self.box_pos, pb.getQuaternionFromEuler((0, 0, self.box_rz)))

    dx = self.box_size[0] / 6
    dy = self.box_size[1] / 4

    pos_candidates = np.array([[3 * dx, -2*dy], [3 * dx, 2*dy],
                               [1.5 * dx, -2 * dy], [1.5 * dx, 2 * dy],
                               [0, -2*dy], [0, 2*dy],
                               [-1.5 * dx, -2 * dy], [-1.5 * dx, 2 * dy],
                               [-3 * dx, -2*dy], [-3 * dx, 2*dy]])

    R = np.array([[np.cos(self.box_rz), -np.sin(self.box_rz)],
                  [np.sin(self.box_rz), np.cos(self.box_rz)]])

    transformed_pos_candidate = R.dot(pos_candidates.T).T
    self.box_placeholder_pos = transformed_pos_candidate + self.box_pos[:2]

  def _getExistingXYPositions(self):
    positions = [o.getXYPosition() for o in self.objects]
    for pos in self.box_placeholder_pos:
      positions.append(list(pos))
    return positions

  def initialize(self):
    super().initialize()
    self.box.initialize(pos=self.box_pos, size=self.box_size)

  def reset(self):
    while True:
      self.resetPybulletEnv()
      self.resetBox()
      try:
        self._generateShapes(constants.RANDOM_BLOCK, self.num_obj, random_orientation=self.random_orientation,
                             padding=self.min_boarder_padding, min_distance=self.min_object_distance)
      except NoValidPositionException:
        continue
      else:
        break
    return self._getObservation()

  def step(self, action):
    self.takeAction(action)
    self.wait(100)
    obs = self._getObservation(action)
    done = self._checkTermination()
    if done:
      reward = self.getReward()
    else:
      reward = 0

    if not done:
      done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
    self.current_episode_steps += 1

    return obs, reward, done

  def _checkTermination(self):
    for obj in self.objects:
      if self.isObjInBox(obj):
        continue
      else:
        return False
    return True

  def getReward(self):
    max_z = max(map(lambda x: pb.getAABB(x.object_id)[1][2], self.objects))
    return 100 * (self.z_threshold - max_z)

  def isObjInBox(self, obj):
    R = np.array([[np.cos(-self.box_rz), -np.sin(-self.box_rz)],
                  [np.sin(-self.box_rz), np.cos(-self.box_rz)]])
    obj_pos = obj.getPosition()[:2]
    transformed_relative_obj_pos = R.dot(np.array([np.array(obj_pos) - self.box_pos[:2]]).T).T[0]

    return -self.box_size[0]/2 < transformed_relative_obj_pos[0] < self.box_size[0]/2 and -self.box_size[1]/2 < transformed_relative_obj_pos[1] < self.box_size[1]/2

  def getObjsOutsideBox(self):
    return list(filter(lambda obj: not self.isObjInBox(obj), self.objects))

  def _getPrimativeHeight(self, motion_primative, x, y):
    if motion_primative == constants.PICK_PRIMATIVE:
      return super()._getPrimativeHeight(motion_primative, x, y)
    else:
      return self.box_size[-1]

def createBlockBinPackingEnv(config):
  return BlockBinPackingEnv(config)

if __name__ == '__main__':
  workspace = np.asarray([[0.2, 0.8],
                          [-0.3, 0.3],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 2, 'action_sequence': 'pxyr', 'num_objects': 9, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'object_init_space_check': 'point', 'physics_mode': 'fast', 'object_scale_range': (0.8, 0.8)}
  planner_config = {'random_orientation': True}

  env = BlockBinPackingEnv(env_config)
  planner = BlockBinPackingPlanner(env, planner_config)
  s, in_hand, obs = env.reset()
  while True:
    action = planner.getNextAction()
    obs, reward, done = env.step(action)
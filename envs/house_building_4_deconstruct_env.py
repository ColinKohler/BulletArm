import time
from copy import deepcopy
import numpy.random as npr
import numpy as np
from itertools import combinations
from helping_hands_rl_envs.envs.pybullet_deconstruct_env import PyBulletEnv, PyBulletDeconstructEnv
from helping_hands_rl_envs.simulators import constants

def createHouseBuilding4DeconstructEnv(simulator_base_env, config):
  class HouseBuilding4DeconstructEnv(PyBulletDeconstructEnv):
    ''''''
    def __init__(self, config):
      if simulator_base_env is PyBulletEnv:
        super().__init__(config)
        self.pick_offset = 0.01
        self.block_scale_range = (0.6, 0.6)
        self.min_block_size = self.block_original_size * self.block_scale_range[0]
        self.max_block_size = self.block_original_size * self.block_scale_range[1]
      else:
        raise ValueError('Bad simulator base env specified.')
      self.simulator_base_env = simulator_base_env
      self.random_orientation = config['random_orientation'] if 'random_orientation' in config else False
      self.num_obj = config['num_objects'] if 'num_objects' in config else 1
      self.reward_type = config['reward_type'] if 'reward_type' in config else 'sparse'

    def step(self, action):
      reward = 1.0 if self.checkStructure() else 0.0
      self.takeAction(action)
      self.wait(100)
      obs = self._getObservation(action)
      motion_primative, x, y, z, rot = self._decodeAction(action)
      done = motion_primative and self._checkTermination()

      if not done:
        done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
      self.current_episode_steps += 1

      return obs, reward, done

    def reset(self):
      ''''''
      while True:
        super(HouseBuilding4DeconstructEnv, self).reset()
        self.generateH4()

        while not self.checkStructure():
          super(HouseBuilding4DeconstructEnv, self).reset()
          self.generateH4()

        return self._getObservation()

    def _checkTermination(self):
      obj_combs = combinations(self.objects, 2)
      for (obj1, obj2) in obj_combs:
        dist = np.linalg.norm(np.array(obj1.getXYPosition()) - np.array(obj2.getXYPosition()))
        if dist < 2.4*self.min_block_size:
          return False
      return True

    def checkStructure(self):
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

      level1_blocks = list(filter(self._isObjOnGround, blocks))
      if len(level1_blocks) != 2:
        return False

      level2_blocks = list(set(blocks) - set(level1_blocks))
      return self._checkOnTop(level1_blocks[0], bricks[0]) and \
             self._checkOnTop(level1_blocks[1], bricks[0]) and \
             self._checkOnTop(bricks[0], level2_blocks[0]) and \
             self._checkOnTop(bricks[0], level2_blocks[1]) and \
             self._checkOnTop(level2_blocks[0], roofs[0]) and \
             self._checkOnTop(level2_blocks[1], roofs[0]) and \
             self._checkOriSimilar([bricks[0], roofs[0]]) and \
             self._checkInBetween(bricks[0], level1_blocks[0], level1_blocks[1]) and \
             self._checkInBetween(roofs[0], level2_blocks[0], level2_blocks[1]) and \
             self._checkInBetween(bricks[0], level2_blocks[0], level2_blocks[1])

    def isSimValid(self):
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
      return self._checkObjUpright(roofs[0]) and super().isSimValid()

  def _thunk():
    return HouseBuilding4DeconstructEnv(config)

  return _thunk
from copy import deepcopy
import pybullet as pb
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.envs.pybullet_tilt_env import PyBulletTiltEnv
from helping_hands_rl_envs.simulators.pybullet.robots.kuka_float_pick import KukaFloatPick
from helping_hands_rl_envs.simulators import constants
import numpy.random as npr
import numpy as np

def createTiltHouseBuilding3Env(simulator_base_env, config):
  class TiltHouseBuilding3Env(PyBulletTiltEnv):
    def __init__(self, config):
      if simulator_base_env is PyBulletEnv:
        super().__init__(config)
      else:
        raise ValueError('Bad simulator base env specified.')
      self.simulator_base_env = simulator_base_env
      self.random_orientation = config['random_orientation'] if 'random_orientation' in config else False
      self.num_obj = config['num_objects'] if 'num_objects' in config else 1
      self.reward_type = config['reward_type'] if 'reward_type' in config else 'sparse'


    def step(self, action):
      self.takeAction(action)
      self.wait(100)
      obs = self._getObservation(action)
      done = self._checkTermination()
      reward = 1.0 if done else 0.0

      if not done:
        done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
      self.current_episode_steps += 1

      return obs, reward, done

    def reset(self):
      # super().reset()
      obj_dict = {
        constants.ROOF: 1,
        constants.BRICK: 1,
        constants.CUBE: 2
      }
      self.resetWithTiltAndObj(obj_dict)
      return self._getObservation()

    def _checkTermination(self):
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

      top_blocks = []
      for block in blocks:
        if self._isObjOnTop(block, blocks):
          top_blocks.append(block)
      if len(top_blocks) != 2:
        return False
      if self._checkOnTop(top_blocks[0], bricks[0]) and \
          self._checkOnTop(top_blocks[1], bricks[0]) and \
          self._checkOnTop(bricks[0], roofs[0]) and \
          self._checkOriSimilar([bricks[0], roofs[0]]) and \
          self._checkInBetween(bricks[0], blocks[0], blocks[1]) and \
          self._checkInBetween(roofs[0], blocks[0], blocks[1]):
        return True
      return False

  def _thunk():
    return TiltHouseBuilding3Env(config)

  return _thunk
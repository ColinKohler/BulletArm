import pybullet as pb
import numpy as np
import scipy
import numpy.random as npr
from copy import deepcopy
from helping_hands_rl_envs.simulators.pybullet.utils import pybullet_util

from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv, NoValidPositionException
from helping_hands_rl_envs.envs.pybullet_drawer_env import PyBulletDrawerEnv
from helping_hands_rl_envs.simulators import constants

def createDrawerOpeningEnv(simulator_base_env, config):
  class DrawerOpeningEnv(PyBulletDrawerEnv):
    def __init__(self, config):
      if simulator_base_env is PyBulletEnv:
        super().__init__(config)
      else:
        raise ValueError('Bad simulator base env specified.')
      self.simulator_base_env = simulator_base_env
      self.random_orientation = config['random_orientation'] if 'random_orientation' in config else False
      if self.random_orientation:
        self.drawer_rot_range = (-np.pi / 2, np.pi / 2)
      else:
        self.drawer_rot_range = (0, 0)
      self.num_obj = config['num_objects'] if 'num_objects' in config else 1
      self.reward_type = config['reward_type'] if 'reward_type' in config else 'sparse'
      self.handle_pos = None

    def reset(self):
      super().reset()
      self.handle_pos = self.drawer.getHandlePosition()
      return self._getObservation()

    def step(self, action):
      motion_primative, x, y, z, rot = self._decodeAction(action)
      action = self._encodeAction(constants.PULL_PRIMATIVE, x, y, z, rot)
      self.takeAction(action)
      self.wait(100)
      obs = self._getObservation(action)
      done = self._checkTermination((x, y, z))
      reward = 1.0 if done else 0.0

      if not done:
        done = self.current_episode_steps >= self.max_steps or not self.isSimValid() or not self.drawer.isDrawerClosed()
      self.current_episode_steps += 1

      return obs, reward, done

    def _getObservation(self, action=None):
      state, in_hand, obs = super(DrawerOpeningEnv, self)._getObservation(action)
      return 0, in_hand, obs

    def _checkTermination(self, action_pos):
      return np.linalg.norm(np.array(action_pos)-np.array(self.handle_pos))<0.05 and self.drawer.isDrawerOpen()


  def _thunk():
    return DrawerOpeningEnv(config)

  return _thunk

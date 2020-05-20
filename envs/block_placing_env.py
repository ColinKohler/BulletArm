from copy import deepcopy
import pybullet as pb
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators.pybullet.robots.kuka_float_pick import KukaFloatPick
from helping_hands_rl_envs.simulators import constants
import numpy.random as npr
import numpy as np

def createBlockPlacingEnv(simulator_base_env, config):
  class BlockPlacingEnv(PyBulletEnv):
    def __init__(self, config):
      config['check_random_obj_valid'] = True
      if simulator_base_env is PyBulletEnv:
        super().__init__(config)
      else:
        raise ValueError('Bad simulator base env specified.')
      self.simulator_base_env = simulator_base_env
      self.random_orientation = config['random_orientation'] if 'random_orientation' in config else False
      self.num_obj = config['num_objects'] if 'num_objects' in config else 1
      self.reward_type = config['reward_type'] if 'reward_type' in config else 'sparse'
      self.rx_range = (-np.pi/4, np.pi/4)
      self.pick_rx = 0

    def pick(self):
      obj = self.objects[0]
      for obj in self.objects:
        if self._isObjOnGround(obj) and self._isObjOnTop(obj):
          break
      pos, rot = obj.getPose()
      rot = self.convertQuaternionToEuler(rot)
      rx = (self.rx_range[1] - self.rx_range[0]) * np.random.random_sample() + self.rx_range[0]
      rz = rot[2]
      self.pick_rx = rx
      pick_action = self._encodeAction(constants.PICK_PRIMATIVE, pos[0], pos[1], pos[2], (rz, rx))
      self.takeAction(pick_action)
      return pick_action

    def step(self, action):
      action[self.action_sequence.find('p')] = 1
      self.takeAction(action)
      self.wait(100)
      done = self._checkTermination()
      reward = 1.0 if done else 0.0
      if not done:
        pick_action = self.pick()
        obs = self._getObservation(pick_action)
      else:
        obs = self._getObservation()

      if not done:
        done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
      self.current_episode_steps += 1

      return obs, reward, done

    def reset(self):
      ''''''
      while True:
        super(BlockPlacingEnv, self).reset()
        try:
          self._generateShapes(constants.CUBE, self.num_obj, random_orientation=self.random_orientation, padding=self.max_block_size*4)
        except:
          continue
        else:
          break
      pick_action = self.pick()
      return self._getObservation(pick_action)

    def _checkTermination(self):
      ''''''
      return self._checkStack()

    def _getObservation(self, action=None):
      state, in_hand, obs = super(BlockPlacingEnv, self)._getObservation(action)
      return 1, in_hand, obs

  def _thunk():
    return BlockPlacingEnv(config)

  return _thunk
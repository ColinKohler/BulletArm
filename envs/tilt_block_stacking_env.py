from copy import deepcopy
import pybullet as pb
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators.pybullet.robots.kuka_float_pick import KukaFloatPick
from helping_hands_rl_envs.simulators import constants
import numpy.random as npr
import numpy as np

def createTiltBlockStackingEnv(simulator_base_env, config):
  class TiltBlockStackingEnv(PyBulletEnv):
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
      self.rx_range = (0, np.pi/4)
      self.tilt_plane_rx = 0
      self.tilt_plane_id = -1
      self.pick_rx = 0

    def step(self, action):
      motion_primative, x, y, z, rot = self._decodeAction(action)
      if motion_primative == constants.PICK_PRIMATIVE:
        self.pick_rx = rot[0]
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
      ''''''
      while True:
        super(TiltBlockStackingEnv, self).reset()
        self.tilt_plane_rx = (self.rx_range[1] - self.rx_range[0]) * np.random.random_sample() + self.rx_range[0]
        self.tilt_plane_id = pb.loadURDF('plane.urdf', [0, -0.1, 0], pb.getQuaternionFromEuler([self.tilt_plane_rx, 0, 0]),
                                 globalScaling=0.01)
        try:
          self._generateShapes(constants.CUBE, self.num_obj, random_orientation=self.random_orientation, padding=self.max_block_size*4)
        except:
          continue
        else:
          break
      return self._getObservation()

    def _checkTermination(self):
      ''''''
      return self._checkStack()

    def _getObservation(self, action=None):
      state, in_hand, obs = super(TiltBlockStackingEnv, self)._getObservation(action)
      return 1, in_hand, obs

  def _thunk():
    return TiltBlockStackingEnv(config)

  return _thunk
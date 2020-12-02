from copy import deepcopy
import pybullet as pb
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators.pybullet.robots.kuka_float_pick import KukaFloatPick
from helping_hands_rl_envs.simulators import constants
import numpy.random as npr
import numpy as np

def createCubeFloatPickingEnv(simulator_base_env, config):
  class CubeFloatPickingEnv(PyBulletEnv):
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
      self.obj_grasped = 0
      self.rx_range = (-np.pi/8, np.pi/8)
      self.robot = KukaFloatPick()

      self.gravity = (0, 0, 0)

    def _getInitializeOrientation(self, random_orientation):
      if random_orientation:
        orientation = pb.getQuaternionFromEuler([(self.rx_range[1]-self.rx_range[0])*np.random.random_sample()+self.rx_range[0],
                                                 0,
                                                 2 * np.pi * np.random.random_sample()])
      else:
        orientation = pb.getQuaternionFromEuler([0., 0., np.pi / 2])
      return orientation

    def step(self, action):
      pre_obj_grasped = self.obj_grasped
      self.takeAction(action)
      self.wait(100)
      obs = self._getObservation(action)
      done = self._checkTermination()
      if self.reward_type == 'dense':
        reward = 1.0 if self.obj_grasped > pre_obj_grasped else 0.0
      else:
        reward = 1.0 if done else 0.0

      if not done:
        done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
      self.current_episode_steps += 1

      return obs, reward, done

    def reset(self):
      ''''''
      while True:
        super(CubeFloatPickingEnv, self).reset()
        try:
          for i in range(self.num_obj):
            self._generateShapes(constants.CUBE, 1, random_orientation=self.random_orientation, padding=self.max_block_size*3)
        except:
          continue
        else:
          break
      self.obj_grasped = 0
      return self._getObservation()

    def saveState(self):
      super(CubeFloatPickingEnv, self).saveState()
      self.state['obj_grasped'] = deepcopy(self.obj_grasped)

    def restoreState(self):
      super(CubeFloatPickingEnv, self).restoreState()
      self.obj_grasped = self.state['obj_grasped']

    def _checkTermination(self):
      ''''''
      for obj in self.objects:
        if self._isObjectHeld(obj):
          self.obj_grasped += 1
          self._removeObject(obj)
          if self.obj_grasped == self.num_obj:
            return True
          return False
      return False

    def _getObservation(self, action=None):
      state, in_hand, obs = super(CubeFloatPickingEnv, self)._getObservation(action)
      return 0, np.zeros_like(in_hand), obs

  def _thunk():
    return CubeFloatPickingEnv(config)

  return _thunk
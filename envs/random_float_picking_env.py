from copy import deepcopy
import pybullet as pb
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants
import numpy.random as npr
import numpy as np

def createRandomFloatPickingEnv(simulator_base_env, config):
  class RandomFloatPickingEnv(PyBulletEnv):
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
      self.obj_succeed = 0
      self.rx_range = (-np.pi/4, np.pi/4)

      pb.setGravity(0, 0, 0)

    def _getInitializeOrientation(self, random_orientation):
      if random_orientation:
        orientation = pb.getQuaternionFromEuler([(self.rx_range[1]-self.rx_range[0])*np.random.random_sample()+self.rx_range[0],
                                                 0,
                                                 2 * np.pi * np.random.random_sample()])
      else:
        orientation = pb.getQuaternionFromEuler([0., 0., np.pi / 2])
      return orientation

    def step(self, action):
      pre_obj_succeed = self.obj_succeed
      self.takeAction(action)
      self.wait(100)
      obs = self._getObservation(action)
      done = self._checkTermination()
      if self.reward_type == 'dense':
        reward = 1.0 if self.obj_succeed > pre_obj_succeed else 0.0
      else:
        reward = 1.0 if self.obj_succeed == self.num_obj else 0.0

      if not done:
        done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
      self.current_episode_steps += 1

      return obs, reward, done

    def reset(self):
      ''''''
      while True:
        super(RandomFloatPickingEnv, self).reset()
        try:
          for i in range(self.num_obj):
            self._generateShapes(constants.RANDOM, 1, random_orientation=self.random_orientation, z_scale=npr.choice([1, 2], p=[0.75, 0.25]))
        except:
          continue
        else:
          break
      self.obj_grasped = 0
      self.obj_succeed = 0
      return self._getObservation()

    def saveState(self):
      super(RandomFloatPickingEnv, self).saveState()
      self.state['obj_grasped'] = deepcopy(self.obj_grasped)

    def restoreState(self):
      super(RandomFloatPickingEnv, self).restoreState()
      self.obj_grasped = self.state['obj_grasped']

    def _checkTermination(self):
      ''''''
      for obj in self.objects:
        if self._isObjectHeld(obj):
          self.obj_grasped += 1
          endToObj = self.robot.getEndToHoldingObj()
          if endToObj is not None and np.abs(endToObj[0, 0]) > 0.98 and np.abs(endToObj[1, 1]) > 0.98 and np.abs(endToObj[2, 2]) > 0.98:
            self.obj_succeed += 1
          self._removeObject(obj)
          if self.obj_grasped == self.num_obj:
            return True
          return False
      return False

    def _getObservation(self, action=None):
      state, in_hand, obs = super(RandomFloatPickingEnv, self)._getObservation(action)
      return 0, in_hand, obs

  def _thunk():
    return RandomFloatPickingEnv(config)

  return _thunk
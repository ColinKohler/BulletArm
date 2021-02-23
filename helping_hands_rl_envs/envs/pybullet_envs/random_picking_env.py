from copy import deepcopy
import numpy as np
from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants
import numpy.random as npr
from helping_hands_rl_envs.simulators.constants import NoValidPositionException

class RandomPickingEnv(PyBulletEnv):
  def __init__(self, config):
    config['check_random_obj_valid'] = True
    super(RandomPickingEnv,  self).__init__(config)
    self.obj_grasped = 0

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
      self.resetPybulletEnv()
      try:
        for i in range(self.num_obj):
          self._generateShapes(constants.RANDOM, 1, random_orientation=self.random_orientation, z_scale=npr.choice([1, 2], p=[0.75, 0.25]))
      except NoValidPositionException:
        continue
      else:
        break
    self.obj_grasped = 0
    return self._getObservation()

  def saveState(self):
    super(RandomPickingEnv, self).saveState()
    self.state['obj_grasped'] = deepcopy(self.obj_grasped)

  def restoreState(self):
    super(RandomPickingEnv, self).restoreState()
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
    state, in_hand, obs = super(RandomPickingEnv, self)._getObservation(action)
    return 0, np.zeros_like(in_hand), obs

def createRandomPickingEnv(config):
  return RandomPickingEnv(config)

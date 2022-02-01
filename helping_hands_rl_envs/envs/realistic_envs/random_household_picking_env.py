import numpy as np
from helping_hands_rl_envs.envs.base_env import BaseEnv
from helping_hands_rl_envs.pybullet.utils import constants
from helping_hands_rl_envs.pybullet.utils.constants import NoValidPositionException

class RandomHouseholdPickingEnv(BaseEnv):
  '''
  '''
  def __init__(self, config):
    # env specific parameters
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.6, 0.6]
    if 'num_objects' not in config:
      config['num_objects'] = 3
    if 'max_steps' not in config:
      config['max_steps'] = 10
    super(RandomHouseholdPickingEnv, self).__init__(config)
    self.obj_grasped = 0
    self.pick_offset = 0.03

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
      self.resetPybulletWorkspace()
      try:
        self._generateShapes(constants.RANDOM_HOUSEHOLD, self.num_obj, random_orientation=self.random_orientation,
                             padding=self.min_boarder_padding, min_distance=self.min_object_distance)
      except NoValidPositionException:
        continue
      else:
        break
    self.obj_grasped = 0
    return self._getObservation()

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
    state, in_hand, obs = super(RandomHouseholdPickingEnv, self)._getObservation()
    return 0, np.zeros_like(in_hand), obs

def createRandomHouseholdPickingEnv(config):
  return RandomHouseholdPickingEnv(config)

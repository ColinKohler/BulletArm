from copy import deepcopy
import numpy as np
from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils import constants

class BlockPickingEnv(BaseEnv):
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
    super(BlockPickingEnv, self).__init__(config)
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
    self.resetPybulletWorkspace()
    self._generateShapes(constants.CUBE, self.num_obj, random_orientation=self.random_orientation)
    self.obj_grasped = 0
    return self._getObservation()

  def saveState(self):
    super(BlockPickingEnv, self).saveState()
    self.state['obj_grasped'] = deepcopy(self.obj_grasped)

  def restoreState(self):
    super(BlockPickingEnv, self).restoreState()
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
    state, in_hand, obs = super(BlockPickingEnv, self)._getObservation()
    return 0, np.zeros_like(in_hand), obs

def createBlockPickingEnv(config):
  return BlockPickingEnv(config)

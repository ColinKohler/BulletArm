import numpy as np
from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils.constants import NoValidPositionException
from bulletarm.pybullet.equipments.container_box import ContainerBox

class RandomBlockPickingClutterEnv(BaseEnv):
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
    super(RandomBlockPickingClutterEnv, self).__init__(config)
    self.object_init_z = 0.1
    self.obj_grasped = 0
    self.box = ContainerBox()

  def initialize(self):
    super().initialize()
    self.box.initialize(pos=[self.workspace[0].mean(), self.workspace[1].mean(), 0],
                        size=[self.workspace_size+0.02, self.workspace_size+0.02, 0.02])


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
        for i in range(self.num_obj):
          self._generateShapes(constants.RANDOM_BLOCK, 1, random_orientation=self.random_orientation,
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
    state, in_hand, obs = super(RandomBlockPickingClutterEnv, self)._getObservation()
    return 0, np.zeros_like(in_hand), obs

def createRandomBlockPickingClutterEnv(config):
  return RandomBlockPickingClutterEnv(config)

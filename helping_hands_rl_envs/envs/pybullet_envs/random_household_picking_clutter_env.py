from copy import deepcopy
import numpy as np
from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.constants import NoValidPositionException
from helping_hands_rl_envs.simulators.pybullet.equipments.box import Box
import pybullet as pb

class RandomHouseholdPickingClutterEnv(PyBulletEnv):
  '''
  '''
  def __init__(self, config):
    super(RandomHouseholdPickingClutterEnv, self).__init__(config)
    self.object_init_z = 0.1
    self.obj_grasped = 0
    self.box = Box()

  def initialize(self):
    super().initialize()
    self.box.initialize(pos=[self.workspace[0].mean(), self.workspace[1].mean(), 0], size=[self.workspace_size+0.02, self.workspace_size+0.02, 0.02])


  def step(self, action):
    pre_obj_grasped = self.obj_grasped
    self.takeAction(action)
    self.wait(100)
    # remove obj that above a threshold hight
    # for obj in self.objects:
    #   if obj.getPosition()[2] > self.pick_pre_offset:
    #     # self.objects.remove(obj)
    #     # pb.removeBody(obj.object_id)
    #     self._removeObject(obj)

    # for obj in self.objects:
    #   if not self._isObjectWithinWorkspace(obj):
    #     self._removeObject(obj)

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
          self._generateShapes(constants.RANDOM_HOUSEHOLD, 1, random_orientation=self.random_orientation,
                               padding=self.min_boarder_padding, min_distance=self.min_object_distance)
      except NoValidPositionException:
        continue
      else:
        break
    self.obj_grasped = 0
    self.num_in_box_obj = self.num_obj
    return self._getObservation()

  def isObjInBox(self, obj_pos, box_pos, box_size):
    box_range = self.box_range(box_pos, box_size)
    return box_range[0][0] < obj_pos[0] < box_range[0][1] and box_range[1][0] < obj_pos[1] < box_range[1][1]

  @staticmethod
  def box_range(box_pos, box_size):
    return np.array([[box_pos[0] - box_size[0] / 2, box_pos[0] + box_size[0] / 2],
                     [box_pos[1] - box_size[1] / 2, box_pos[1] + box_size[1] / 2]])

  def InBoxObj(self, box_pos, box_size):
    obj_list = []
    for obj in self.objects:
      if self.isObjInBox(obj.getPosition(), box_pos, box_size):
        obj_list.append(obj)
    return obj_list

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

  # def _checkTermination(self):
  #   ''''''
  #   self.num_in_box_obj = len(self.InBoxObj([self.workspace[0].mean(), self.workspace[1].mean(), 0],
  #                                      [self.workspace_size+0.02, self.workspace_size+0.02, 0.02]))
  #   return self.num_in_box_obj == 0

  def _getObservation(self, action=None):
    state, in_hand, obs = super(RandomHouseholdPickingClutterEnv, self)._getObservation()
    return 0, np.zeros_like(in_hand), obs

def createRandomHouseholdPickingClutterEnv(config):
  return RandomHouseholdPickingClutterEnv(config)

from copy import deepcopy
import numpy as np
from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.constants import NoValidPositionException
from helping_hands_rl_envs.simulators.pybullet.equipments.tray import Tray
import pybullet as pb
import os
import pybullet_data

class RandomHouseholdPickingClutterEnv(PyBulletEnv):
  '''
  '''
  def __init__(self, config):
    super(RandomHouseholdPickingClutterEnv, self).__init__(config)
    self.object_init_z = 0.1
    self.obj_grasped = 0
    self.tray = Tray()

  def initialize(self):
    super().initialize()
    self.tray.initialize(pos=[self.workspace[0].mean(), self.workspace[1].mean(), 0],
                         size=[self.workspace_size+0.015, self.workspace_size+0.015, 0.1])


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
        # self.trayUid = pb.loadURDF(os.path.join(pybullet_data.getDataPath(), "tray/tray.urdf"),
        #                            self.workspace[0].mean(), self.workspace[1].mean(), 0,
        #                            0.000000, 0.000000, 1.000000, 0.000000)
        for i in range(self.num_obj):
          x = (np.random.rand() - 0.5) * 0.1
          x += self.workspace[0].mean()
          y = (np.random.rand() - 0.5) * 0.1
          y += self.workspace[1].mean()
          randpos = [x, y, 0.20]
          obj = self._generateShapes(constants.RANDOM_HOUSEHOLD, 1, random_orientation=self.random_orientation,
                                     pos=[randpos], padding=self.min_boarder_padding,
                                     min_distance=self.min_object_distance)
          pb.changeDynamics(obj[0].object_id, -1, lateralFriction=0.6)
      except NoValidPositionException:
        continue
      else:
        break
    self.obj_grasped = 0
    self.num_in_tray_obj = self.num_obj
    return self._getObservation()

  def isObjInBox(self, obj_pos, tray_pos, tray_size):
    tray_range = self.tray_range(tray_pos, tray_size)
    return tray_range[0][0] < obj_pos[0] < tray_range[0][1] and tray_range[1][0] < obj_pos[1] < tray_range[1][1]

  @staticmethod
  def tray_range(tray_pos, tray_size):
    return np.array([[tray_pos[0] - tray_size[0] / 2, tray_pos[0] + tray_size[0] / 2],
                     [tray_pos[1] - tray_size[1] / 2, tray_pos[1] + tray_size[1] / 2]])

  def InBoxObj(self, tray_pos, tray_size):
    obj_list = []
    for obj in self.objects:
      if self.isObjInBox(obj.getPosition(), tray_pos, tray_size):
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
  #   self.num_in_tray_obj = len(self.InBoxObj([self.workspace[0].mean(), self.workspace[1].mean(), 0],
  #                                      [self.workspace_size+0.02, self.workspace_size+0.02, 0.02]))
  #   return self.num_in_tray_obj == 0

  def _getObservation(self, action=None):
    state, in_hand, obs = super(RandomHouseholdPickingClutterEnv, self)._getObservation()
    return 0, np.zeros_like(in_hand), obs

def createRandomHouseholdPickingClutterEnv(config):
  return RandomHouseholdPickingClutterEnv(config)

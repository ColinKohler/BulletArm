import os
import pybullet as pb
import copy
import numpy as np
import numpy.random as npr

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.equipments.box_color import BoxColor
from helping_hands_rl_envs.simulators.pybullet.equipments.rack import Rack
from helping_hands_rl_envs.simulators.pybullet.objects.plate import PLACE_RY_OFFSET, PLACE_Z_OFFSET
from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.utils import transformations
from helping_hands_rl_envs.simulators.constants import NoValidPositionException
from helping_hands_rl_envs.planners.shelf_bowl_stacking_planner import ShelfBowlStackingPlanner
from helping_hands_rl_envs.simulators.pybullet.utils import pybullet_util

class CovidTestEnv(PyBulletEnv):
  def __init__(self, config):
    super().__init__(config)
    self.object_init_space = np.array([[0.3, 0.7],
                                       [-0.4, 0],
                                       [0, 0.40]])
    self.plate_model_id = None
    self.place_ry_offset = None
    self.end_effector_santilized_t = 0

    # for workspace = 0.4, flexible
    # rot90 x n
    R0 = np.array([[1., 0.],
                   [0., 1.]])
    R1 = np.array([[0., -1.],
                   [1., 0.]])
    R2 = np.array([[-1., 0.],
                   [0., -1.]])
    R3 = np.array([[0., 1.],
                   [-1., 0.]])
    self.rot90 = [R0, R1, R2, R3]
    # flip transform
    Nf = R0
    Vf = np.array([[-1., 0.],
                   [0., 1.]])
    self.flips = [Nf, Vf]

    self.boxs = [BoxColor(), BoxColor(), BoxColor(), BoxColor()]
    self.new_tube_box_size = np.array([0.225, 0.225, 0.005])
    # self.santilizing_box_size = np.array([0.225, 0.07, 0.035])
    self.used_tube_box_size = np.array([0.225, 0.095, 0.03])
    self.test_box_size = np.array([0.165, 0.4, 0.01])

    workspace_x = self.workspace[0, 1] - self.workspace[0, 0]
    workspace_y = self.workspace[1, 1] - self.workspace[1, 0]
    center_x = (self.workspace[0, 0] + self.workspace[0, 1]) / 2
    center_y = (self.workspace[1, 0] + self.workspace[1, 1]) / 2
    self.workspace_center = np.array([center_x, center_y, 0])
    upper_x = -workspace_x / 2 + self.new_tube_box_size[0] / 2
    lower_x = workspace_x / 2 - self.test_box_size[0] / 2
    new_tube_box_y = workspace_y / 2 - self.new_tube_box_size[1] / 2
    used_tube_box_y = -workspace_y / 2 + self.used_tube_box_size[1] / 2
    # santilizing_box_y = (new_tube_box_y - self.new_tube_box_size[1] / 2 +
    #                     used_tube_box_y + self.used_tube_box_size[1] / 2) / 2
    self.new_tube_box_pos_o = np.array([upper_x, new_tube_box_y, 0])
    self.used_tube_box_pos_o = np.array([upper_x, used_tube_box_y, 0])
    # self.santilizing_box_pos_o = np.array([upper_x, santilizing_box_y, 0])
    self.test_box_pos_o = np.array([lower_x, 0, 0])

    self.tube_pos_candidate_o = np.array([[-self.new_tube_box_size[0]/3.5, self.new_tube_box_size[1]/4, 0.05],
                                       [                           0, self.new_tube_box_size[1]/4, 0.05],
                                       [ self.new_tube_box_size[0]/3.5, self.new_tube_box_size[1]/4, 0.05]])
    self.swab_pos_candidate_o = np.array([[-self.new_tube_box_size[0]/3.5, -self.new_tube_box_size[1]/9, 0.02],
                                       [                           0, -self.new_tube_box_size[1]/9, 0.02],
                                       [ self.new_tube_box_size[0]/3.5, -self.new_tube_box_size[1]/9, 0.02]])
    self.tube_pos_candidate_o += self.new_tube_box_pos_o
    self.swab_pos_candidate_o += self.new_tube_box_pos_o


  def initialize(self):
    super().initialize()
    self.robot.gripper_joint_limit = [0, 0.15]
    self.boxs[0].initialize(pos=self.new_tube_box_pos_o + self.workspace_center,
                        size=self.new_tube_box_size, color=[0.9, 0.9, 1, 1])
    self.boxs[1].initialize(pos=self.test_box_pos_o + self.workspace_center,
                        size=self.test_box_size, color=[0.9, 0.9, 0.9, 1])
    # self.boxs[2].initialize(pos=self.santilizing_box_pos_o + self.workspace_center,
    #                     size=self.santilizing_box_size, color=[0.5, 0.5, 0.5, 0.6])
    self.boxs[3].initialize(pos=self.used_tube_box_pos_o + self.workspace_center,
                        size=self.used_tube_box_size, color=[1, 1, 0.5, 1])
    pass

  def resetBoxs(self):

    # initial boxs in D4 group: rot90 + reflection
    rot_n = np.random.randint(0,4)
    flip = np.random.randint(0,2)
    self.flip = flip
    # rot_n = 0
    # flip = 0

    self.new_tube_box_pos = np.zeros((3))
    self.used_tube_box_pos = np.zeros((3))
    # self.santilizing_box_pos = np.zeros((3))
    self.test_box_pos = np.zeros((3))
    self.new_tube_box_pos[:2] = self.flips[flip].dot(self.rot90[rot_n].dot(self.new_tube_box_pos_o[:2].T)).T
    self.used_tube_box_pos[:2] = self.flips[flip].dot(self.rot90[rot_n].dot(self.used_tube_box_pos_o[:2].T)).T
    # self.santilizing_box_pos[:2] = self.flips[flip].dot(self.rot90[rot_n].dot(self.santilizing_box_pos_o[:2].T)).T
    self.test_box_pos[:2] = self.flips[flip].dot(self.rot90[rot_n].dot(self.test_box_pos_o[:2].T)).T

    self.new_tube_box_pos += self.workspace_center
    self.used_tube_box_pos += self.workspace_center
    # self.santilizing_box_pos += self.workspace_center
    self.test_box_pos += self.workspace_center

    self.rot90x = np.pi*rot_n/2
    self.rot_n = rot_n
    self.boxs[0].reset(list(self.new_tube_box_pos), pb.getQuaternionFromEuler((0,0,self.rot90x)))
    self.boxs[1].reset(list(self.test_box_pos), pb.getQuaternionFromEuler((0,0,self.rot90x)))
    # self.boxs[2].reset(list(self.santilizing_box_pos), pb.getQuaternionFromEuler((0,0,self.rot90x)))
    self.boxs[3].reset(list(self.used_tube_box_pos), pb.getQuaternionFromEuler((0,0,self.rot90x)))

    self.tube_pos_candidate = 0.01 * np.ones_like(self.tube_pos_candidate_o)
    self.swab_pos_candidate = 0.01 * np.ones_like(self.swab_pos_candidate_o)
    self.tube_pos_candidate[:,:2] = self.flips[flip].dot(self.rot90[rot_n].dot(self.tube_pos_candidate_o[:,:2].T)).T
    self.swab_pos_candidate[:,:2] = self.flips[flip].dot(self.rot90[rot_n].dot(self.swab_pos_candidate_o[:,:2].T)).T

    self.tube_pos_candidate += self.workspace_center
    self.swab_pos_candidate += self.workspace_center

  def reset(self):
    # self.end_effector_santilized_t = 0
    self.placed_swab = False
    self.resetted = True
    self.no_obj_split = True
    self.swab_delet_times = 0
    self.used_tube = None
    self.num_tubes_in_used_box = 0

    while True:
      self.resetPybulletEnv()
      self.resetBoxs()
      try:
        tube_rot = 1.57 + np.random.rand() - 0.5 + self.rot90x + np.random.randint(2) * np.pi
        for i in range(3):
          self._generateShapes(constants.TEST_TUBE,
                               rot=[pb.getQuaternionFromEuler([0., 0., tube_rot])],
                               pos=[tuple(self.tube_pos_candidate[i])])
        if self.flip == 1:
          swab_rot = np.pi - (-1.57 + np.random.rand() - 0.5 + self.rot90x)
        else:
          swab_rot = -1.57 + np.random.rand() - 0.5 + self.rot90x
        for i in range(3):
          self._generateShapes(constants.SWAB,
                               rot=[pb.getQuaternionFromEuler([0., 0., swab_rot])],
                               pos=[tuple(self.swab_pos_candidate[i])])
      except NoValidPositionException:
        continue
      else:
        break

    self._getObservation()
    return self._getObservation()

  def step(self, action):
    self.takeAction(action)
    motion_primative, x, y, z, rot = self._decodeAction(action)
    # if self.rot_n % 2 == 1:
    #   rot_santilizing_box_size = [self.santilizing_box_size[1], self.santilizing_box_size[0]]
    # else:
    #   rot_santilizing_box_size = self.santilizing_box_size[:2]
    # if motion_primative == 0 and \
    #         self.isObjInBox([x, y, z], self.santilizing_box_pos, rot_santilizing_box_size):
    #   self.end_effector_santilized_t = 1
    # elif self.end_effector_santilized_t > 0:
    #   self.end_effector_santilized_t += 1
    on_table_obj, on_table_obj_type = self.OnTableObj()
    num_on_table_tube = 0
    num_on_table_swab = 0
    for obj in on_table_obj:
      if obj.object_type_id == constants.SWAB:
        num_on_table_swab += 1
      if obj.object_type_id == constants.TEST_TUBE:
        num_on_table_tube += 1
    # if num_on_table_tube == 0 and num_on_table_swab == 0:
    #   self.swab_delet_times = 0

    if num_on_table_tube == 1 and num_on_table_swab == 1 and self.used_tube is None:
      for obj in on_table_obj:
        if obj.object_type_id == constants.SWAB:
          self.objects.remove(obj)
          pb.removeBody(obj.object_id)
        if obj.object_type_id == constants.TEST_TUBE:
          if self.rot_n % 2 == 1:
            rot_test_box_size = [self.test_box_size[1], self.test_box_size[0]]
          else:
            rot_test_box_size = self.test_box_size[:2]
          rot = 2 * np.pi * np.random.rand()
          x_offset = (rot_test_box_size[0] - 0.08) * np.random.rand()\
                     - (rot_test_box_size[0] - 0.08) / 2
          y_offset = (rot_test_box_size[1] - 0.08) * np.random.rand()\
                     - (rot_test_box_size[1] - 0.08) / 2
          obj_rot_ = pb.getQuaternionFromEuler([0, 0, rot])
          obj.resetPose(self.test_box_pos + [x_offset, y_offset, 0.05], obj_rot_)
          self.used_tube = obj
    elif num_on_table_tube == 1 and num_on_table_swab == 1 and self.used_tube:
      num_on_table_tube = 2  # end the episode

    # if constants.TEST_TUBE in on_table_obj_type \
    #   and constants.SWAB in on_table_obj_type:
    #   for obj in on_table_obj:
    #     if num_on_table_tube > 1 or num_on_table_swab > 1:
    #       break
    #     if obj.object_type_id == constants.SWAB and self.swab_delet_times == 0 and not self.used_tube:
    #       self.objects.remove(obj)
    #       pb.removeBody(obj.object_id)
    #       self.swab_delet_times += 1
    #       self.used_tube = True
    #     if obj.object_type_id == constants.TEST_TUBE and self.swab_delet_times == 0:
    #       if self.rot_n % 2 == 1:
    #         rot_test_box_size = [self.test_box_size[1], self.test_box_size[0]]
    #       else:
    #         rot_test_box_size = self.test_box_size[:2]
    #       rot = 2 * np.pi * np.random.rand()
    #       x_offset = (rot_test_box_size[0] - 0.08) * np.random.rand()\
    #                  - (rot_test_box_size[0] - 0.08) / 2
    #       y_offset = (rot_test_box_size[1] - 0.08) * np.random.rand()\
    #                  - (rot_test_box_size[1] - 0.08) / 2
    #       obj_rot_ = pb.getQuaternionFromEuler([0, 0, rot])
    #       obj.resetPose(self.test_box_pos + [x_offset, y_offset, 0.05], obj_rot_)
    #       self.used_tube = max(self.used_tube, 1)
    #
    #     self.wait(20)
    #     self.placed_swab = True

    if self.used_tube is not None and self.used_tube in self.tubesInUsedBox():
      self.used_tube = None
    self.wait(100)
    obs = self._getObservation(action)
    done = self._checkTermination()
    if done:
      reward = 1
    else:
      reward = 0

    if not done:
      done = self.current_episode_steps >= self.max_steps\
             or not self.isSimValid()\
             or num_on_table_tube > 1\
             or num_on_table_swab > 1
    self.current_episode_steps += 1

    return obs, reward, done

  def isObjInBox(self, obj_pos, box_pos, box_size):
    box_range = self.box_range(box_pos, box_size)
    return box_range[0][0] < obj_pos[0] < box_range[0][1] and box_range[1][0] < obj_pos[1] < box_range[1][1]

  @staticmethod
  def box_range(box_pos, box_size):
    return np.array([[box_pos[0] - box_size[0] / 2, box_pos[0] + box_size[0] / 2],
                     [box_pos[1] - box_size[1] / 2, box_pos[1] + box_size[1] / 2]])

  def _checkTermination(self):
    # if self.end_effector_santilized_t == 1\
    #         and len(self.tubesInUsedBox()) == 3\
    #         and len(self.numSwabs()) == 0:
    if len(self.tubesInUsedBox()) == 3\
            and len(self.numSwabs()) == 0:
      return True
    return False

  def OnTableObj(self):
    obj_list = []
    obj_type_list = []
    if self.rot_n % 2 == 1:
      rot_test_box_size = [self.test_box_size[1], self.test_box_size[0]]
    else:
      rot_test_box_size = self.test_box_size[:2]
    for obj in self.objects:
      if self.isObjInBox(obj.getPosition(), self.test_box_pos, rot_test_box_size):
        obj_list.append(obj)
        obj_type_list.append(self.object_types[obj])
    return obj_list, obj_type_list

  def InBoxObj(self, box_pos, box_size):
    obj_list = []
    for obj in self.objects:
      if self.isObjInBox(obj.getPosition(), box_pos, box_size):
        obj_list.append(obj)
    return obj_list

  def getSwabs(self):
    swabs = []
    if self.rot_n % 2 == 1:
      rot_new_tube_box_size = [self.new_tube_box_size[1], self.new_tube_box_size[0]]
    else:
      rot_new_tube_box_size = self.new_tube_box_size[:2]
    for obj in self.InBoxObj(self.new_tube_box_pos, rot_new_tube_box_size):
      if obj.object_type_id == constants.SWAB:
        swabs.append(obj)
    return swabs

  def getTubes(self):
    tubes = []
    if self.rot_n == 1 or self.rot_n == 3:
      rot_new_tube_box_size = [self.new_tube_box_size[1], self.new_tube_box_size[0]]
    else:
      rot_new_tube_box_size = self.new_tube_box_size[:2]
    for obj in self.InBoxObj(self.new_tube_box_pos, rot_new_tube_box_size):
      if obj.object_type_id == constants.TEST_TUBE:
        tubes.append(obj)
    return tubes

  def tubesInUsedBox(self):
    tubes = []
    if self.rot_n == 1 or self.rot_n == 3:
      rot_used_tube_box_size = [self.used_tube_box_size[1], self.used_tube_box_size[0]]
    else:
      rot_used_tube_box_size = self.used_tube_box_size[:2]
    for obj in self.InBoxObj(self.used_tube_box_pos, rot_used_tube_box_size):
      if obj.object_type_id == constants.TEST_TUBE:
        tubes.append(obj)
    return tubes

  def numSwabs(self):
    swabs = []
    for obj in self.objects:
      if obj.object_type_id == constants.SWAB:
        swabs.append(obj)
    return swabs


def createCovidTestEnv(config):
  return CovidTestEnv(config)


if __name__ == '__main__':
  object_init_space = np.asarray([[0.3, 0.7],
                          [-0.4, 0.4],
                          [0, 0.40]])
  workspace_size = 0.4
  workspace = np.asarray([[0.5 - workspace_size / 2, 0.5 + workspace_size / 2],
                          [0 - workspace_size / 2, 0 + workspace_size / 2],
                          [0, 0 + workspace_size]])
  env_config = {'workspace': workspace, 'object_init_space': object_init_space, 'max_steps': 10, 'obs_size': 128,
                'render': True, 'fast_mode': True, 'seed': 0, 'action_sequence': 'pxyzrrr', 'num_objects': 9,
                'random_orientation': True, 'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False,
                'object_scale_range': (0.6, 0.6), 'robot': 'kuka', 'object_init_space_check': 'point', 'physics_mode': 'slow'}
  planner_config = {'random_orientation': True}

  env = CovidTestEnv(env_config)
  # planner = ShelfBowlStackingPlanner(env, planner_config)
  s, in_hand, obs = env.reset()
  a = 1
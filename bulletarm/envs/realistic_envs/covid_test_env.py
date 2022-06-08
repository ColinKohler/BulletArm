import pybullet as pb
import numpy as np
import numpy.random as npr

from bulletarm.pybullet.equipments.box_color import BoxColor
from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils.constants import NoValidPositionException

class CovidTestEnv(BaseEnv):
  '''Open loop covid test task.

  The robot needs to supervise three covid tests and gather the test tubes.

  Args:
    config (dict): Intialization arguments for the env
  '''
  def __init__(self, config):
    # env specific parameters
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.55, 0.55]
    if 'num_objects' not in config:
      config['num_objects'] = 6
    if 'max_steps' not in config:
      config['max_steps'] = 30
    super().__init__(config)
    self.object_init_space = np.array([[0.3, 0.7],
                                       [-0.4, 0],
                                       [0, 0.40]])
    self.plate_model_id = None
    self.place_ry_offset = None
    self.end_effector_santilized_t = 0
    self.robot.gripper_joint_limit = [0, 0.08]

    # # for workspace = 0.4, flexible
    # # rot90 x n
    # R0 = np.array([[1., 0.],
    #                [0., 1.]])
    # R1 = np.array([[0., -1.],
    #                [1., 0.]])
    # R2 = np.array([[-1., 0.],
    #                [0., -1.]])
    # R3 = np.array([[0., 1.],
    #                [-1., 0.]])
    # self.rot90 = [R0, R1, R2, R3]
    # flip transform
    Nf = np.array([[1., 0.],
                   [0., 1.]])
    Vf = np.array([[-1., 0.],
                   [0., 1.]])
    self.flips = [Nf, Vf]

    workspace_x = self.workspace[0, 1] - self.workspace[0, 0]
    workspace_y = self.workspace[1, 1] - self.workspace[1, 0]
    self.workspace_padding = 0.05
    center_x = (self.workspace[0, 0] + self.workspace[0, 1]) / 2
    center_y = (self.workspace[1, 0] + self.workspace[1, 1]) / 2
    self.workspace_center = np.array([center_x, center_y, 0])

    self.boxs = [BoxColor(), BoxColor(), BoxColor(), BoxColor()]
    self.new_tube_box_size = np.array([0.17, 0.21, 0.005])
    # self.santilizing_box_size = np.array([0.2, 0.07, 0.035])
    self.used_tube_box_size = np.array([0.12, 0.08, 0.04])
    self.test_box_size = np.array([0.125, workspace_y - 2 * self.workspace_padding, 0.01])

    new_tube_box_x = -(workspace_x / 2 - self.workspace_padding) + self.new_tube_box_size[0] / 2
    used_tube_box_x = -(workspace_x / 2 - self.workspace_padding) + self.used_tube_box_size[0] / 2
    test_box_x = (workspace_x / 2 - self.workspace_padding) - self.test_box_size[0] / 2
    new_tube_box_y = (workspace_y / 2 - self.workspace_padding) - self.new_tube_box_size[1] / 2
    used_tube_box_y = -(workspace_y / 2 - self.workspace_padding) + self.used_tube_box_size[1] / 2
    # santilizing_box_y = (new_tube_box_y - self.new_tube_box_size[1] / 2 +
    #                     used_tube_box_y + self.used_tube_box_size[1] / 2) / 2
    self.new_tube_box_pos_o = np.array([new_tube_box_x, new_tube_box_y, 0])
    self.used_tube_box_pos_o = np.array([used_tube_box_x, used_tube_box_y, 0])
    # self.santilizing_box_pos_o = np.array([upper_x, santilizing_box_y, 0])
    self.test_box_pos_o = np.array([test_box_x, 0, 0])

    self.tube_pos_candidate_o = np.array([[-self.new_tube_box_size[0]/3.3, self.new_tube_box_size[1]/4, 0.05],
                                       [                           0, self.new_tube_box_size[1]/4, 0.05],
                                       [ self.new_tube_box_size[0]/3.3, self.new_tube_box_size[1]/4, 0.05]])
    self.swab_pos_candidate_o = np.array([[-self.new_tube_box_size[0]/3.7, -self.new_tube_box_size[1]/9, 0.02],
                                       [                           0, -self.new_tube_box_size[1]/9, 0.02],
                                       [ self.new_tube_box_size[0]/3.7, -self.new_tube_box_size[1]/9, 0.02]])
    self.tube_pos_candidate_o += self.new_tube_box_pos_o
    self.swab_pos_candidate_o += self.new_tube_box_pos_o

  def perterb_rot90(self, n):
    '''
    generate rotation matrix for rotating 90*n + perterbation degree
    :param n:
    :return:
    '''
    assert n in (0, 1, 2, 3)
    perterb_range = np.arcsin(self.workspace_padding
                              / ((self.workspace[0, 1] - self.workspace[0, 0]) / 2 - self.workspace_padding))
    perterb_range = min(perterb_range, perterb_range - np.pi / 180)
    if self.random_orientation:
      angel = n * (np.pi / 2) + np.random.uniform(-perterb_range, perterb_range)
    else:
      angel = n * (np.pi / 2)
    R_perterb_90 = np.array([[np.cos(angel), -np.sin(angel)],
                             [np.sin(angel),  np.cos(angel)]])
    return R_perterb_90, angel

  def initialize(self):
    super().initialize()
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
    if self.random_orientation:
      self.rot_n = np.random.randint(0,4)
    else:
      self.rot_n = np.random.choice([0, 2])
    self.flip = np.random.randint(0,2)
    self.R_perterb_90, self.R_angel = self.perterb_rot90(self.rot_n)
    self.R_angel_after_flip = np.pi - self.R_angel if self.flip else self.R_angel
    self.perterb_angel = self.R_angel % (np.pi / 2)
    self.perterb_angel = self.perterb_angel if self.perterb_angel < np.pi/4 else np.pi/2 - self.perterb_angel
    T_range = (self.workspace_padding - (np.sin(self.perterb_angel) *
                                         ((self.workspace[0, 1] - self.workspace[0, 0]) / 2 - self.workspace_padding)))
    T_range = min(T_range, T_range - 0.005)
    self.T_perterb_x = np.random.uniform(-1,1) * T_range
    self.T_perterb_y = np.random.uniform(-1,1) * T_range
    # rot_n = 0
    # flip = 0
    self.perterbed_workspace_center = self.workspace_center + np.array([self.T_perterb_x, self.T_perterb_y, 0])

    self.new_tube_box_pos = np.zeros((3))
    self.used_tube_box_pos = np.zeros((3))
    # self.santilizing_box_pos = np.zeros((3))
    self.test_box_pos = np.zeros((3))
    self.new_tube_box_pos[:2] = self.flips[self.flip].dot(self.R_perterb_90.dot(self.new_tube_box_pos_o[:2].T)).T
    self.used_tube_box_pos[:2] = self.flips[self.flip].dot(self.R_perterb_90.dot(self.used_tube_box_pos_o[:2].T)).T
    # self.santilizing_box_pos[:2] = self.flips[self.flip].dot(self.R_perterb_90.dot(self.santilizing_box_pos_o[:2].T)).T
    self.test_box_pos[:2] = self.flips[self.flip].dot(self.R_perterb_90.dot(self.test_box_pos_o[:2].T)).T

    self.new_tube_box_pos += self.perterbed_workspace_center
    self.used_tube_box_pos += self.perterbed_workspace_center
    # self.santilizing_box_pos += self.perterbed_workspace_center
    self.test_box_pos += self.perterbed_workspace_center

    self.boxs[0].reset(list(self.new_tube_box_pos), pb.getQuaternionFromEuler((0,0,self.R_angel_after_flip)))
    self.boxs[1].reset(list(self.test_box_pos), pb.getQuaternionFromEuler((0,0,self.R_angel_after_flip)))
    # self.boxs[2].reset(list(self.santilizing_box_pos), pb.getQuaternionFromEuler((0,0,self.R_angel_after_flip)))
    self.boxs[3].reset(list(self.used_tube_box_pos), pb.getQuaternionFromEuler((0,0,self.R_angel_after_flip)))

    self.tube_pos_candidate = 0.01 * np.ones_like(self.tube_pos_candidate_o)
    self.swab_pos_candidate = 0.01 * np.ones_like(self.swab_pos_candidate_o)
    self.tube_pos_candidate[:,:2] = self.flips[self.flip].dot(self.R_perterb_90.dot(self.tube_pos_candidate_o[:,:2].T)).T
    self.swab_pos_candidate[:,:2] = self.flips[self.flip].dot(self.R_perterb_90.dot(self.swab_pos_candidate_o[:,:2].T)).T

    self.tube_pos_candidate += self.perterbed_workspace_center
    self.swab_pos_candidate += self.perterbed_workspace_center

  def reset(self):
    # self.end_effector_santilized_t = 0
    self.placed_swab = False
    self.resetted = True
    self.no_obj_split = True
    self.swab_delet_times = 0
    self.used_tube = None
    self.num_tubes_in_used_box = 0

    while True:
      self.resetPybulletWorkspace()
      self.resetBoxs()
      try:
        if self.random_orientation:
          tube_rot = 1.57 + np.random.uniform(-np.pi/6, np.pi/6) + self.R_angel_after_flip + np.random.randint(2) * np.pi
        else:
          tube_rot = 1.57 + self.R_angel_after_flip + np.random.randint(2) * np.pi
        for i in range(3):
          self._generateShapes(constants.TEST_TUBE,
                               rot=[pb.getQuaternionFromEuler([0., 0., tube_rot])],
                               pos=[tuple(self.tube_pos_candidate[i])])
        if self.flip == 1:
          if self.random_orientation:
            swab_rot = np.pi - (-1.57 + np.random.uniform(-np.pi/6, np.pi/6) + self.R_angel)
          else:
            swab_rot = np.pi - (-1.57 + self.R_angel)
        else:
          if self.random_orientation:
            swab_rot = -1.57 + np.random.uniform(-np.pi/6, np.pi/6) + self.R_angel
          else:
            swab_rot = -1.57 + self.R_angel
        # swab_rot = np.pi - (-1.57 + np.random.rand() - 0.5 + self.R_angel_after_flip)
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
    on_table_obj, on_table_obj_type = self.OnTableObj()
    num_on_table_tube = 0
    num_on_table_swab = 0
    for obj in on_table_obj:
      if obj.object_type_id == constants.SWAB:
        num_on_table_swab += 1
      if obj.object_type_id == constants.TEST_TUBE:
        num_on_table_tube += 1

    if num_on_table_tube == 1 and num_on_table_swab == 1 and self.used_tube is None:
      for obj in on_table_obj:
        if obj.object_type_id == constants.SWAB:
          self.objects.remove(obj)
          pb.removeBody(obj.object_id)
        if obj.object_type_id == constants.TEST_TUBE:
          # if self.rot_n % 2 == 1:
          #   rot_test_box_size = [self.test_box_size[1], self.test_box_size[0]]
          # else:
          #   rot_test_box_size = self.test_box_size[:2]
          rot = 2 * np.pi * np.random.rand()
          if not self.random_orientation:
            rot = np.pi/2
          x_offset = (self.test_box_size[0] - 0.1) * np.random.rand()\
                     - (self.test_box_size[0] - 0.1) / 2
          y_offset = (self.test_box_size[1] - 0.1) * np.random.rand()\
                     - (self.test_box_size[1] - 0.1) / 2
          obj_rot_ = pb.getQuaternionFromEuler([0, 0, rot])
          used_tube_placing_pose = self.test_box_pos.copy()
          used_tube_placing_pose[:2] += self.flips[self.flip].dot(self.R_perterb_90.dot(np.array([x_offset, y_offset])))
          used_tube_placing_pose[2] = 0.01
          obj.resetPose(used_tube_placing_pose, obj_rot_)
          self.used_tube = obj
    elif num_on_table_tube == 1 and num_on_table_swab == 1 and self.used_tube:
      num_on_table_tube = 2  # end the episode


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

  def isObjInBox(self, obj, box):
    pos, rot = pb.getBasePositionAndOrientation(box.id)
    pos = np.array(pos[:2])
    rot = pb.getEulerFromQuaternion(rot)[2]
    obj_pos, _ = pb.getBasePositionAndOrientation(obj.object_id)
    obj_pos = np.array(obj_pos[:2])
    obj_pos -= pos
    R_inv = np.array([[np.cos(-rot), -np.sin(-rot)],
                      [np.sin(-rot),  np.cos(-rot)]])
    obj_pos = R_inv.dot(obj_pos)
    return np.abs(obj_pos[0]) < box.size[0] / 2 and np.abs(obj_pos[1]) < box.size[1] / 2

  # @staticmethod
  # def box_range(box_pos, box_size):
  #   return np.array([[box_pos[0] - box_size[0] / 2, box_pos[0] + box_size[0] / 2],
  #                    [box_pos[1] - box_size[1] / 2, box_pos[1] + box_size[1] / 2]])

  def _checkTermination(self):
    if len(self.tubesInUsedBox()) == 3\
            and len(self.numSwabs()) == 0:
      return True
    return False

  def OnTableObj(self):
    obj_list = []
    obj_type_list = []
    # if self.rot_n % 2 == 1:
    #   rot_test_box_size = [self.test_box_size[1], self.test_box_size[0]]
    # else:
    #   rot_test_box_size = self.test_box_size[:2]
    for obj in self.objects:
      if self.isObjInBox(obj, self.boxs[1]):
        obj_list.append(obj)
        obj_type_list.append(self.object_types[obj])
    return obj_list, obj_type_list

  def InBoxObj(self, box):
    obj_list = []
    for obj in self.objects:
      if self.isObjInBox(obj, box):
        obj_list.append(obj)
    return obj_list

  def getSwabs(self):
    swabs = []
    # if self.rot_n % 2 == 1:
    #   rot_new_tube_box_size = [self.new_tube_box_size[1], self.new_tube_box_size[0]]
    # else:
    #   rot_new_tube_box_size = self.new_tube_box_size[:2]
    for obj in self.InBoxObj(self.boxs[0]):
      if obj.object_type_id == constants.SWAB:
        swabs.append(obj)
    return swabs

  def getTubes(self):
    tubes = []
    # if self.rot_n == 1 or self.rot_n == 3:
    #   rot_new_tube_box_size = [self.new_tube_box_size[1], self.new_tube_box_size[0]]
    # else:
    #   rot_new_tube_box_size = self.new_tube_box_size[:2]
    for obj in self.InBoxObj(self.boxs[0]):
      if obj.object_type_id == constants.TEST_TUBE:
        tubes.append(obj)
    return tubes

  def tubesInUsedBox(self):
    tubes = []
    # if self.rot_n == 1 or self.rot_n == 3:
    #   rot_used_tube_box_size = [self.used_tube_box_size[1], self.used_tube_box_size[0]]
    # else:
    #   rot_used_tube_box_size = self.used_tube_box_size[:2]
    for obj in self.InBoxObj(self.boxs[3]):
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

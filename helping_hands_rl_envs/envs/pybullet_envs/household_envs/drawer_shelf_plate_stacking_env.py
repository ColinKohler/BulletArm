import os
import pybullet as pb
import copy
import numpy as np
import numpy.random as npr

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.equipments.shelf import Shelf
from helping_hands_rl_envs.simulators.pybullet.equipments.rack import Rack
from helping_hands_rl_envs.simulators.pybullet.equipments.drawer import Drawer
from helping_hands_rl_envs.simulators.pybullet.equipments.drawer_with_rack import DrawerWithRack
from helping_hands_rl_envs.simulators.pybullet.objects.plate import PLACE_RY_OFFSET, PLACE_Z_OFFSET
from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.utils import transformations
from helping_hands_rl_envs.simulators.constants import NoValidPositionException
from helping_hands_rl_envs.planners.shelf_bowl_stacking_planner import ShelfBowlStackingPlanner

class DrawerShelfPlateStackingEnv(PyBulletEnv):
  def __init__(self, config):
    super().__init__(config)
    self.shelf = Shelf()
    # self.rack = Rack(n=self.num_obj+1)
    # self.drawer = Drawer()
    self.drawer_pos = [0.8, -0.2, 0]
    self.drawer_with_rack = DrawerWithRack(self.num_obj + 1)

    self.object_init_space = np.asarray([[0.3, 0.7],
                                         [-0.4, 0],
                                         [0, 0.40]])
    self.plate_model_id = None
    self.place_offset = None
    self.place_ry_offset = None

  def initialize(self):
    super().initialize()
    self.drawer_with_rack.initialize(pos=self.drawer_pos)
    self.shelf.initialize(pos=[0.6, 0.3, 0])
    if self.physic_mode == 'slow':
      pb.changeDynamics(self.shelf.id, -1, linearDamping=0.04, angularDamping=0.04, restitution=0,
                        contactStiffness=3000, contactDamping=100)
    # self.rack.initialize(pos=[0.3, -0.3, 0], rot=transformations.quaternion_from_euler(0, 0, 0))
    # if self.physic_mode == 'slow':
    #   for rack_id in self.rack.ids:
    #     pb.changeDynamics(rack_id, -1, linearDamping=0.04, angularDamping=0.04, restitution=0,
    #                       contactStiffness=3000, contactDamping=100)
    self.robot.gripper_joint_limit = [0, 0.12]
    pass

  def reset(self):
    ''''''
    self.plate_model_id = 0
    self.place_ry_offset = PLACE_RY_OFFSET[self.plate_model_id]
    self.place_offset = PLACE_Z_OFFSET[self.plate_model_id]
    while True:
      self.resetPybulletEnv()
      self.drawer_with_rack.reset(pos=self.drawer_pos)
      try:
        plate_pos_list = self.drawer_with_rack.rack.getObjInitPosList()
        for pos in plate_pos_list:
          self._generateShapes(constants.PLATE, 1, pos=[pos], rot=[transformations.quaternion_from_euler(0, -np.deg2rad(95), 0)], model_id=self.plate_model_id)
      except NoValidPositionException as e:
        continue
      else:
        break

    self.drawer_with_rack.drawer.constraintObjects(self.objects)
    return self._getObservation()

  def step(self, action):
    self.takeAction(action)
    self.wait(100)
    obs = self._getObservation(action)
    done = self._checkTermination()
    reward = 1.0 if done else 0.0

    if not done:
      done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
    self.current_episode_steps += 1

    if self.drawer_with_rack.drawer.isDrawerOpen():
      self.drawer_with_rack.drawer.releaseObjectConstraints()

    return obs, reward, done

  def getPlaceRyOffset(self):
    return PLACE_RY_OFFSET[self.plate_model_id]

  def anyObjectOnTarget1(self):
    for obj in self.objects:
      if self.shelf.isObjectOnTarget1(obj):
        return True
    return False

  def _checkTermination(self):
    return self._checkStack() and self.anyObjectOnTarget1()

  def _getValidOrientation(self, random_orientation):
    if random_orientation:
      orientation = pb.getQuaternionFromEuler([0., 0., np.pi * -(np.random.random_sample() * 0.5 + 0.5)])
    else:
      orientation = pb.getQuaternionFromEuler([0., 0., 0.])
    return orientation
    
  def getValidSpace(self):
    return self.object_init_space

  def test(self):
    handle_pos = self.drawer_with_rack.drawer.getHandlePosition()
    rot = pb.getQuaternionFromEuler((0, -np.pi/2, 0))
    self.robot.pull(handle_pos, rot, 0.3)


def createDrawerShelfPlateStackingEnv(config):
  return DrawerShelfPlateStackingEnv(config)
  

if __name__ == '__main__':
  object_init_space = np.asarray([[0.3, 0.7],
                          [-0.4, 0.4],
                          [0, 0.40]])
  env_config = {'object_init_space': object_init_space, 'max_steps': 10, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyzrrr', 'num_objects': 2, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'object_init_space_check': 'point', 'physics_mode': 'slow'}
  planner_config = {'random_orientation': True}

  env = DrawerShelfPlateStackingEnv(env_config)
  planner = ShelfBowlStackingPlanner(env, planner_config)
  s, in_hand, obs = env.reset()
  env.test()
  print(1)
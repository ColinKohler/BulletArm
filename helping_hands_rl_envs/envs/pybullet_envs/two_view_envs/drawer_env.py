import os
import pybullet as pb
import numpy as np

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.equipments.drawer import Drawer
from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants

class DrawerEnv(PyBulletEnv):
  def __init__(self, config):
    super().__init__(config)
    self.drawer_theta_range = [-np.pi/6, np.pi/6]
    self.drawer1 = Drawer()
    self.drawer2 = Drawer()
    self.drawer_theta = 0

    self.wall_x = 1.2
    self.wall_id = None

    self.drawer_pos_variation = 0.1

  def resetDrawerEnv(self):
    self.resetPybulletEnv()

    if self.random_orientation:
      self.drawer_theta = np.random.random() * (self.drawer_theta_range[1] - self.drawer_theta_range[0]) + self.drawer_theta_range[0]
    else:
      self.drawer_theta = 0
    drawer_rotq = pb.getQuaternionFromEuler((0, 0, self.drawer_theta))

    drawer_pos_x = self.workspace[0].mean() + 0.41 - 0.05
    drawer_pos_y = 0

    drawer_pos_x += ((np.random.random()-0.5) * self.drawer_pos_variation)
    drawer_pos_y += ((np.random.random()-0.5) * self.drawer_pos_variation)

    self.drawer1.reset((drawer_pos_x, drawer_pos_y, 0), drawer_rotq)
    self.drawer2.reset((drawer_pos_x, drawer_pos_y, 0.36*0.5), drawer_rotq)

  def reset(self):
    self.resetDrawerEnv()
    return self._getObservation()

  def initialize(self):
    super().initialize()

    self.drawer1.initialize((self.workspace[0].mean() + 0.41, 0, 0), pb.getQuaternionFromEuler((0, 0, 0)))
    self.drawer2.initialize((self.workspace[0].mean() + 0.41, 0, 0.36*0.5), pb.getQuaternionFromEuler((0, 0, 0)))

    root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
    urdf_filepath = os.path.join(root_dir, 'simulators/urdf/', 'wall.urdf')
    self.wall_id = pb.loadURDF(urdf_filepath,
                               [self.wall_x,
                                self.workspace[1].mean(),
                                0],
                               pb.getQuaternionFromEuler([0, 0, 0]),
                               globalScaling=1)

  def isSimValid(self):
    for obj in self.objects:
      if self.drawer1.isObjInsideDrawer(obj) or self.drawer2.isObjInsideDrawer(obj):
        continue
      if not self.check_random_obj_valid and self.object_types[obj] == constants.RANDOM:
        continue
      if self._isObjectHeld(obj):
        continue
      p = obj.getPosition()
      if self.workspace_check == 'point':
        if not self._isPointInWorkspace(p):
          return False
      else:
        if not self._isObjectWithinWorkspace(obj):
          return False
      if self.pos_candidate is not None:
        if np.abs(self.pos_candidate[0] - p[0]).min() > 0.02 or np.abs(self.pos_candidate[1] - p[1]).min() > 0.02:
          return False
    return True

  def test(self):
    handle1_pos = self.drawer1.getHandlePosition()
    handle2_pos = self.drawer2.getHandlePosition()
    rot = pb.getQuaternionFromEuler((0, -np.pi/2, 0))
    self.robot.pull(handle1_pos, rot, 0.2)
    self.robot.pull(handle2_pos, rot, 0.2)


if __name__ == '__main__':
  workspace = np.asarray([[0.3, 0.7],
                          [-0.2, 0.2],
                          [0, 0.40]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyrr', 'num_objects': 5, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point'}
  env = DrawerEnv(env_config)
  while True:
    s, in_hand, obs = env.reset()
    # env.test()
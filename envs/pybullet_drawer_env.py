import pybullet as pb
import numpy as np
import copy
import scipy
import numpy.random as npr
import matplotlib.pyplot as plt

from helping_hands_rl_envs.simulators.pybullet.equipments.drawer import Drawer
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv, NoValidPositionException
import helping_hands_rl_envs.simulators.pybullet.utils.object_generation as pb_obj_generation
from helping_hands_rl_envs.simulators import constants

class PyBulletDrawerEnv(PyBulletEnv):
  def __init__(self, config):
    super().__init__(config)
    self.drawer = Drawer()
    self.drawer_rot_range = (0,0)

  def initializeDrawer(self, rot_range=(-np.pi/2,np.pi/2)):
    rot = np.random.random()*(rot_range[1]-rot_range[0])+rot_range[0]
    dist_drawer2center = self.workspace_size/2 + 0.1
    x = self.workspace[0].mean() + dist_drawer2center * np.cos(rot)
    y = self.workspace[1].mean() + dist_drawer2center * np.sin(rot)
    self.drawer.initialize((x, y, 0), pb.getQuaternionFromEuler((0, 0, rot)))

  def initialize(self):
    super().initialize()
    self.initializeDrawer(self.drawer_rot_range)
    # self.drawer.initialize((self.workspace[0].max()+0.1, self.workspace[1].mean(), 0.2))
    # pb_obj_generation.generateCube((self.workspace[0].max()-0.05, self.workspace[1].mean(), 0.05), pb.getQuaternionFromEuler((0, 0, 0)), 0.6)

  def reset(self):
    if not self.initialized:
      self.initialize()
      self.initialized = True
    self.episode_count += 1
    if self.episode_count >= 1000:
      self.initialize()
      self.episode_count = 0

    for o in self.objects:
      pb.removeBody(o.object_id)
    self.drawer.remove()

    self.robot.reset()
    self.initializeDrawer(self.drawer_rot_range)
    self.objects = list()
    self.object_types = {}
    self.heightmap = None
    self.current_episode_steps = 1
    self.last_action = None
    self.last_obj = None
    self.state = {}
    self.pb_state = None

    while True:
      try:
        self._generateShapes(constants.RANDOM, self.num_random_objects, random_orientation=True)
      except Exception as e:
        continue
      else:
        break

    pb.stepSimulation()

    return self._getObservation()


  def test(self):
    handle_pos = self.drawer.getHandlePosition()
    state, in_hand, obs = self._getObservation()
    handle_pos = list(handle_pos)
    # handle_pos[0] -= 0.02
    # handle_pos[2] -= 0.04
    # pre_pos = copy.copy(handle_pos)
    # after_pos = copy.copy(handle_pos)
    # pre_pos[2] += 0.1
    # after_pos[0] -= 0.2
    # self.robot.moveTo(pre_pos, pb.getQuaternionFromEuler((0, -np.pi / 6, 0)))
    # self.robot.moveTo(handle_pos, pb.getQuaternionFromEuler((0, -np.pi / 6, 0)))
    # self.robot.closeGripper()
    # self.robot.moveTo(after_pos, pb.getQuaternionFromEuler((0, -np.pi / 6, 0)))

    rot = pb.getQuaternionFromEuler((0, -np.pi/2, 0))
    self.robot.pull(handle_pos, rot, 0.2)

    pass

if __name__ == '__main__':
  workspace = np.asarray([[0.3, 0.7],
                          [-0.2, 0.2],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyrr', 'num_objects': 5, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point'}

  env = PyBulletDrawerEnv(env_config)
  while True:
    env.reset()
    env.test()

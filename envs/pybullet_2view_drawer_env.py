import pybullet as pb
import numpy as np
import copy
import scipy
import os
import helping_hands_rl_envs
import numpy.random as npr
import matplotlib.pyplot as plt

from helping_hands_rl_envs.simulators.pybullet.equipments.drawer import Drawer
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv, NoValidPositionException
from helping_hands_rl_envs.envs.pybullet_2view_env import PyBullet2ViewEnv
import helping_hands_rl_envs.simulators.pybullet.utils.object_generation as pb_obj_generation
from helping_hands_rl_envs.simulators import constants

class PyBullet2ViewDrawerEnv(PyBullet2ViewEnv):
  def __init__(self, config):
    super().__init__(config)
    self.drawer = Drawer()
    self.drawer2 = Drawer()


  def reset(self):
    super().reset()

    self.drawer.remove()
    self.drawer2.remove()
    self.drawer.initialize((self.workspace[0][1]+0.11, 0, 0), pb.getQuaternionFromEuler((0, 0, 0)))
    self.drawer2.initialize((self.workspace[0][1]+0.11, 0, 0.18), pb.getQuaternionFromEuler((0, 0, 0)))

    return self._getObservation()

  def test(self):
    handle1_pos = self.drawer.getHandlePosition()
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
  env = PyBullet2ViewEnv(env_config)
  while True:
    s, in_hand, obs = env.reset()
    env.test()
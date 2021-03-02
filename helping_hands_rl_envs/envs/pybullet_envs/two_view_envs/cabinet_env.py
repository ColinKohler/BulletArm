import os
import pybullet as pb
import numpy as np

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.equipments.cabinet import Cabinet
from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.utils import transformations

class CabinetEnv(PyBulletEnv):
  def __init__(self, config):
    super().__init__(config)
    self.cabinet = Cabinet()

  def resetCabinetEnv(self):
    pass

  def reset(self):
    self.resetPybulletEnv()
    self.resetCabinetEnv()
    return self._getObservation()

  def initialize(self):
    super().initialize()
    self.cabinet.initialize((self.workspace[0].mean() + 0.2, 0, 1.35), pb.getQuaternionFromEuler((0, 0, np.pi)))
    pass

  def test(self):
    # self.robot.roundPull(self.cabinet.getLeftHandlePos(), pb.getQuaternionFromEuler((-np.pi/2, 0, np.pi/2)), 0.2, 0.29, left=True, dynamic=True)

    self.robot.pull(self.cabinet.getLeftHandlePos(), pb.getQuaternionFromEuler((-np.pi/2, 0, np.pi/2)), 0.2, dynamic=True)
    push_pos = self.cabinet.getLeftHandlePos()
    push_pos[0] += 0.05
    self.robot.push(push_pos, pb.getQuaternionFromEuler((np.pi/10, -np.pi/2, 0)), 0.2, dynamic=False)
    pass

  def _isObjOnTop(self, obj, objects=None):
    if not objects:
      objects = self.objects
    obj_position = obj.getPosition()
    for o in objects:
      if self._isObjectHeld(o) or o is obj:
        continue
      block_position = o.getPosition()
      if obj.isTouching(o) and block_position[-1] > obj_position[-1]:
        return False
    return True


if __name__ == '__main__':
  workspace = np.asarray([[0.3, 0.7],
                          [-0.2, 0.2],
                          [0, 0.40]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyzrrr', 'num_objects': 5, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point', 'physics_mode': 'slow'}
  env = CabinetEnv(env_config)
  s, in_hand, obs = env.reset()
  env.test()
import pybullet as pb
import numpy as np

from helping_hands_rl_envs.simulators.pybullet.equipments.drawer import Drawer
from helping_hands_rl_envs.envs.pybullet_envs.two_view_envs.two_view_env import TwoViewEnv
from helping_hands_rl_envs.simulators import constants

class TwoViewDrawerEnv(TwoViewEnv):
  def __init__(self, config):
    super().__init__(config)
    self.drawer1 = Drawer()
    self.drawer2 = Drawer()

  def reset(self):
    super().reset()

    self.drawer1.reset()
    self.drawer2.reset()

    return self._getObservation()

  def initialize(self):
    super().initialize()
    # self.drawer.remove()
    # self.drawer2.remove()
    self.drawer1.initialize((self.workspace[0][1] + 0.21, 0, 0), pb.getQuaternionFromEuler((0, 0, 0)))
    self.drawer2.initialize((self.workspace[0][1] + 0.21, 0, 0.18), pb.getQuaternionFromEuler((0, 0, 0)))

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
  env = TwoViewDrawerEnv(env_config)
  while True:
    s, in_hand, obs = env.reset()
    env.test()
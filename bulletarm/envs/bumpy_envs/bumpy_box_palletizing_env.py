import pybullet as pb
import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils.constants import NoValidPositionException

from bulletarm.envs.realistic_envs.box_palletizing_env import BoxPalletizingEnv
from bulletarm.envs.bumpy_envs.bumpy_base import BumpyBase
from bulletarm.planners.box_palletizing_planner import BoxPalletizingPlanner

class BumpyBoxPalletizingEnv(BoxPalletizingEnv, BumpyBase):
  def __init__(self, config):
    BoxPalletizingEnv.__init__(self, config)
    BumpyBase.__init__(self)
    self.pallet_z += self.bump_offset
    self.place_offset += self.bump_offset

  def initialize(self):
    BoxPalletizingEnv.initialize(self)
    BumpyBase.initialize(self)

  # def reset(self):
  #   BumpyBase.resetBumps(self)
  #   re = BoxPalletizingEnv.reset(self)
  #   BumpyBase.resetPlatform(self, self.pallet_pos, self.pallet_rz, self.pallet_size)
  #   return re

  def generateOneBox(self):
    super().generateOneBox()
    self._changeBoxDynamics(self.objects[-1])
    for _ in range(100):
      pb.stepSimulation()

  def reset(self):
    while True:
      if self.pallet is not None:
        pb.removeBody(self.pallet.object_id)
      self.resetPybulletWorkspace()
      BumpyBase.resetBumps(self)
      self.resetPallet()
      pb.changeDynamics(self.pallet.object_id, -1, linearDamping=0.04, angularDamping=0.04, restitution=0,
                        contactStiffness=3000, contactDamping=100)
      BumpyBase.resetPlatform(self, self.pallet_pos, self.pallet_rz, self.pallet_size)

      try:
        self._generateShapes(constants.BOX, 1, random_orientation=self.random_orientation)
        self._changeBoxDynamics(self.objects[-1])
      except NoValidPositionException:
        continue
      else:
        break
    for _ in range(100):
      pb.stepSimulation()
    return self._getObservation()

  def getObjEachLevel(self):
    level1_threshold = self.pallet_z + self.pallet_height/2 + 0.5 * self.box_height - 0.01
    level2_threshold = self.pallet_z + self.pallet_height/2 + 1.5 * self.box_height - 0.01
    level3_threshold = self.pallet_z + self.pallet_height/2 + 2.5 * self.box_height - 0.01
    level4_threshold = self.pallet_z + self.pallet_height/2 + 3.5 * self.box_height - 0.01
    level1_objs = list(filter(lambda o: level1_threshold < o.getZPosition() < level2_threshold, self.objects))
    level2_objs = list(filter(lambda o: level2_threshold < o.getZPosition() < level3_threshold, self.objects))
    level3_objs = list(filter(lambda o: level3_threshold < o.getZPosition() < level4_threshold, self.objects))
    return level1_objs, level2_objs, level3_objs

  def _isObjOnGround(self, obj):
    contact_points = obj.getContactPoints()
    for p in contact_points:
      if p[2] == self.table_id or p[2] in self.bump_ids:
        return True
    return False

  def _changeBoxDynamics(self, box):
    pb.changeDynamics(box.object_id, -1, linearDamping=0.04, angularDamping=0.04, restitution=0,
                      contactStiffness=3000, contactDamping=100)

def createBumpyBoxPalletizingEnv(config):
  return BumpyBoxPalletizingEnv(config)

if __name__ == '__main__':
  workspace = np.asarray([[0.3, 0.7],
                          [-0.2, 0.2],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 36, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyzrrr', 'num_objects': 18, 'random_orientation': True,
                'reward_type': 'sparse', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point', 'physics_mode': 'fast', 'hard_reset_freq': 1000, 'object_scale_range': (0.8, 0.8),
                'kuka_adjust_gripper_offset': 0.001,
                }

  planner_config = {'random_orientation': True, 'half_rotation': True}

  env = BumpyBoxPalletizingEnv(env_config)
  planner = BoxPalletizingPlanner(env, planner_config)
  s, in_hand, obs = env.reset()
  while True:
    action = planner.getNextAction()
    obs, reward, done = env.step(action)

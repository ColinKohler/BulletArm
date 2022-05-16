import pybullet as pb
import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils.constants import NoValidPositionException

from bulletarm.envs.block_structure_envs.house_building_4_env import HouseBuilding4Env
from bulletarm.envs.bumpy_envs.bumpy_base import BumpyBase
from bulletarm.planners.bumpy_house_building_4_planner import BumpyHouseBuilding4Planner

class BumpyHouseBuilding4Env(HouseBuilding4Env, BumpyBase):
  def __init__(self, config):
    HouseBuilding4Env.__init__(self, config)
    BumpyBase.__init__(self)
    # self.place_offset = self.bump_offset

    self.platform_pos = self.workspace[:2].mean(1)
    self.platform_size = 0.13

  def initialize(self):
    HouseBuilding4Env.initialize(self)
    BumpyBase.initialize(self)

  def reset(self):
    ''''''
    while True:
      self.resetPybulletWorkspace()
      self.platform_pos = self._getValidPositions(self.platform_size*2, 0, [], 1)[0]
      BumpyBase.resetBumps(self)
      BumpyBase.resetPlatform(self, self.platform_pos, np.random.random() * np.pi, [self.platform_size, self.platform_size])

      try:
        self._generateShapes(constants.ROOF, 1, random_orientation=self.random_orientation)
        self._generateShapes(constants.BRICK, 1, random_orientation=self.random_orientation)
        self._generateShapes(constants.CUBE, 4, random_orientation=self.random_orientation)
        for _ in range(100):
          pb.stepSimulation()
        if not self.isSimValid():
          continue
      except NoValidPositionException as e:
        continue
      else:
        break
    return self._getObservation()

  def _getExistingXYPositions(self):
    positions = [o.getXYPosition() for o in self.objects]
    pt_placeholder = np.array(np.meshgrid(np.linspace(-0.08, 0.08, 5), np.linspace(-0.08, 0.08, 5))).T.reshape(-1, 2) + self.platform_pos
    positions.extend(pt_placeholder.tolist())
    return positions

  def _checkTermination(self):
    blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
    bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

    level1_blocks = list(filter(self.isObjOnPlatform, blocks))
    if len(level1_blocks) != 2:
      return False

    level2_blocks = list(set(blocks) - set(level1_blocks))
    return self._checkOnTop(level1_blocks[0], bricks[0]) and \
           self._checkOnTop(level1_blocks[1], bricks[0]) and \
           self._checkOnTop(bricks[0], level2_blocks[0]) and \
           self._checkOnTop(bricks[0], level2_blocks[1]) and \
           self._checkOnTop(level2_blocks[0], roofs[0]) and \
           self._checkOnTop(level2_blocks[1], roofs[0]) and \
           self._checkOriSimilar([bricks[0], roofs[0]]) and \
           self._checkInBetween(bricks[0], level1_blocks[0], level1_blocks[1]) and \
           self._checkInBetween(roofs[0], level2_blocks[0], level2_blocks[1]) and \
           self._checkInBetween(bricks[0], level2_blocks[0], level2_blocks[1])

def createBumpyHouseBuilding4Env(config):
  return BumpyHouseBuilding4Env(config)

if __name__ == '__main__':
  workspace = np.asarray([[0.3, 0.7],
                          [-0.2, 0.2],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 36, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyzrrr', 'num_objects': 18, 'random_orientation': True,
                'reward_type': 'sparse', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point', 'physics_mode': 'fast', 'hard_reset_freq': 1000,
                'object_scale_range': (0.6, 0.6),
                }

  planner_config = {'random_orientation': True, 'half_rotation': True}

  env = BumpyHouseBuilding4Env(env_config)
  planner = BumpyHouseBuilding4Planner(env, planner_config)
  s, in_hand, obs = env.reset()
  while True:
    action = planner.getNextAction()
    obs, reward, done = env.step(action)

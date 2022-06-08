import pybullet as pb
import numpy as np
import numpy.random as npr

from bulletarm.pybullet.equipments.container_box import ContainerBox
from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils.constants import NoValidPositionException
from bulletarm.planners.block_bin_packing_planner import BlockBinPackingPlanner

class BlockBinPackingEnv(BaseEnv):
  '''Open loop bin packing task.

  The robot needs to pack the N blocks in the workspace inside a bin.
  The number of blocks N is set by the config.

  Args:
    config (dict): Intialization arguments for the env
  '''
  def __init__(self, config):
    # env specific parameters
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.8, 0.8]
    if 'num_objects' not in config:
      config['num_objects'] = 8
    if 'max_steps' not in config:
      config['max_steps'] = 20
    if 'min_object_distance' not in config:
      config['min_object_distance'] = 0.09
    if 'min_boarder_padding' not in config:
      config['min_boarder_padding'] = 0.05

    super().__init__(config)
    self.box = ContainerBox()
    self.box_rz = 0
    self.box_pos = [0.60, 0.12, 0]
    self.box_size = [0.23*self.block_scale_range[1], 0.15*self.block_scale_range[1], 0.1]
    self.box_placeholder_pos = []
    self.z_threshold = self.box_size[-1]

  def resetBox(self):
    self.box_rz = np.random.random_sample() * np.pi if self. random_orientation else 0
    self.box_pos = self._getValidPositions(np.linalg.norm([self.box_size[0], self.box_size[1]]), 0, [], 1)[
      0]
    self.box_pos.append(0)
    self.box.reset(self.box_pos, pb.getQuaternionFromEuler((0, 0, self.box_rz)))

    dx = self.box_size[0] / 6
    dy = self.box_size[1] / 4

    pos_candidates = np.array([[3 * dx, -2*dy], [3 * dx, 2*dy],
                               [1.5 * dx, -2 * dy], [1.5 * dx, 2 * dy],
                               [0, -2*dy], [0, 2*dy],
                               [-1.5 * dx, -2 * dy], [-1.5 * dx, 2 * dy],
                               [-3 * dx, -2*dy], [-3 * dx, 2*dy]])

    R = np.array([[np.cos(self.box_rz), -np.sin(self.box_rz)],
                  [np.sin(self.box_rz), np.cos(self.box_rz)]])

    transformed_pos_candidate = R.dot(pos_candidates.T).T
    self.box_placeholder_pos = transformed_pos_candidate + self.box_pos[:2]

  def _getExistingXYPositions(self):
    positions = [o.getXYPosition() for o in self.objects]
    for pos in self.box_placeholder_pos:
      positions.append(list(pos))
    return positions

  def initialize(self):
    super().initialize()
    self.box.initialize(pos=self.box_pos, size=self.box_size, thickness=0.002)

  def reset(self):
    while True:
      self.resetPybulletWorkspace()
      self.resetBox()
      try:
        self._generateShapes(constants.RANDOM_BLOCK, self.num_obj, random_orientation=self.random_orientation,
                             padding=self.min_boarder_padding, min_distance=self.min_object_distance)
      except NoValidPositionException:
        continue
      else:
        break
    return self._getObservation()

  def _checkTermination(self):
    for obj in self.objects:
      if self.isObjInBox(obj):
        continue
      else:
        return False
    return True

  def isObjInBox(self, obj):
    R = np.array([[np.cos(-self.box_rz), -np.sin(-self.box_rz)],
                  [np.sin(-self.box_rz), np.cos(-self.box_rz)]])
    obj_pos = obj.getPosition()[:2]
    transformed_relative_obj_pos = R.dot(np.array([np.array(obj_pos) - self.box_pos[:2]]).T).T[0]

    return -self.box_size[0]/2 < transformed_relative_obj_pos[0] < self.box_size[0]/2 and -self.box_size[1]/2 < transformed_relative_obj_pos[1] < self.box_size[1]/2

  def getObjsOutsideBox(self):
    return list(filter(lambda obj: not self.isObjInBox(obj), self.objects))

  def _getPrimativeHeight(self, motion_primative, x, y):
    if motion_primative == constants.PICK_PRIMATIVE:
      return super()._getPrimativeHeight(motion_primative, x, y)
    else:
      return self.box_size[-1]

def createBlockBinPackingEnv(config):
  return BlockBinPackingEnv(config)

if __name__ == '__main__':
  workspace = np.asarray([[0.2, 0.8],
                          [-0.3, 0.3],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 2, 'action_sequence': 'pxyr', 'num_objects': 9, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'object_init_space_check': 'point', 'physics_mode': 'fast', 'object_scale_range': (0.8, 0.8)}
  planner_config = {'random_orientation': True}

  env = BlockBinPackingEnv(env_config)
  planner = BlockBinPackingPlanner(env, planner_config)
  s, in_hand, obs = env.reset()
  while True:
    action = planner.getNextAction()
    obs, reward, done = env.step(action)

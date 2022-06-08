import pybullet as pb
import numpy as np

from bulletarm.pybullet.equipments.container_box import ContainerBox
from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils.constants import NoValidPositionException
from bulletarm.planners.bottle_tray_planner import BottleTrayPlanner

class BottleTrayEnv(BaseEnv):
  '''Open loop bottle arrangement task.

  The robot needs to arrange six bottle in the tray.

  Args:
    config (dict): Intialization arguments for the env
  '''
  def __init__(self, config):
    # env specific parameters
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.8, 0.8]
    if 'num_objects' not in config:
      config['num_objects'] = 6
    if 'max_steps' not in config:
      config['max_steps'] = 20
    if 'kuka_adjust_gripper_offset' not in config:
      config['kuka_adjust_gripper_offset'] = 0.0025
    super().__init__(config)
    self.place_offset = 0.2*self.block_scale_range[1]
    self.box = ContainerBox()
    self.box_rz = 0
    self.box_size = [0.1*self.block_scale_range[1]*3, 0.1*self.block_scale_range[1]*2, 0.05]
    self.box_pos = [0.4, 0.2, 0]
    self.place_pos_candidate = []
    pass

  def resetBox(self):
    if not self.random_orientation:
      self.box_rz = 0
    else:
      self.box_rz = np.random.random_sample() * np.pi
    self.box_pos = self._getValidPositions(np.linalg.norm([self.box_size[0]/3, self.box_size[1]/4])*2, 0, [], 1)[0]
    self.box_pos.append(0)
    self.box.reset(self.box_pos, pb.getQuaternionFromEuler((0, 0, self.box_rz)))

    dx = self.box_size[0]/6
    dy = self.box_size[1]/4

    pos_candidates = np.array([[2*dx, -dy], [2*dx, dy],
                               [0, -dy], [0, dy],
                               [-2*dx, -dy], [-2*dx, dy]])

    R = np.array([[np.cos(self.box_rz), -np.sin(self.box_rz)],
                  [np.sin(self.box_rz), np.cos(self.box_rz)]])

    transformed_pos_candidate = R.dot(pos_candidates.T).T
    self.place_pos_candidate = transformed_pos_candidate + self.box_pos[:2]

    pass

  def _getExistingXYPositions(self):
    positions = [o.getXYPosition() for o in self.objects]
    for pos in self.place_pos_candidate:
      positions.append(list(pos))
    return positions

  def initialize(self):
    super().initialize()
    self.box.initialize(pos=self.box_pos, size=self.box_size)


  def reset(self):
    while True:
      self.resetPybulletWorkspace()
      self.resetBox()
      try:
        self._generateShapes(constants.BOTTLE, self.num_obj, random_orientation=self.random_orientation, padding=self.min_boarder_padding, min_distance=self.min_object_distance)
      except NoValidPositionException:
        continue
      else:
        break

    return self._getObservation()

  def step(self, action):
    motion_primative, x, y, z, rot = self._decodeAction(action)
    if motion_primative == constants.PICK_PRIMATIVE:
      self.place_offset = z + 0.005
    return super().step(action)

  def _checkTermination(self):
    for obj in self.objects:
      if self.isObjInBox(obj) and self._checkObjUpright(obj):
        continue
      else:
        return False
    return True

  def isSimValid(self):
    for obj in self.objects:
      if not self._checkObjUpright(obj):
        return False
    return super().isSimValid()

  def isObjInBox(self, obj):
    return obj.isTouchingId(self.box.id)

  def getObjsOutsideBox(self):
    return list(filter(lambda obj: not self.isObjInBox(obj), self.objects))

def createBottleTrayEnv(config):
  return BottleTrayEnv(config)

if __name__ == '__main__':
  workspace = np.asarray([[0.2, 0.8],
                          [-0.3, 0.3],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 2, 'action_sequence': 'pxyr', 'num_objects': 6, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'object_init_space_check': 'point', 'physics_mode': 'fast', 'object_scale_range': (1, 1)}
  planner_config = {'random_orientation': True}

  env = BottleTrayEnv(env_config)
  planner = BottleTrayPlanner(env, planner_config)
  s, in_hand, obs = env.reset()
  while True:
    action = planner.getNextAction()
    obs, reward, done = env.step(action)

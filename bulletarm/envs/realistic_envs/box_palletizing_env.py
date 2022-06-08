import pybullet as pb
import numpy as np

from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils import transformations
from bulletarm.pybullet.utils.constants import NoValidPositionException
from bulletarm.pybullet.objects.pallet import Pallet
from bulletarm.planners.box_palletizing_planner import BoxPalletizingPlanner

class BoxPalletizingEnv(BaseEnv):
  '''Open loop box palletizing task.

  The robot needs to palletize N boxes on top of a pallet.
  The number of blocks N is set by the config.

  Args:
    config (dict): Intialization arguments for the env
  '''
  def __init__(self, config):
    # env specific parameters
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.8, 0.8]
    if 'num_objects' not in config:
      config['num_objects'] = 18
    if 'max_steps' not in config:
      config['max_steps'] = 40
    if 'kuka_adjust_gripper_offset' not in config:
      config['kuka_adjust_gripper_offset'] = 0.0025
    super().__init__(config)
    self.pallet_height = 0.04625*self.block_scale_range[1]
    self.pallet_z = 0
    self.pallet_pos = [0.5, 0.1, self.pallet_height/2]
    self.pallet_size = [np.mean(self.block_scale_range)*0.074*3, np.mean(self.block_scale_range)*0.108*2]
    self.pallet_rz = 0
    self.pallet = None
    self.box_height = 0.056 * self.block_scale_range[1]

    self.pick_offset = 0.02 * self.block_scale_range[1]
    self.place_offset = 0.06 * self.block_scale_range[1]

    self.odd_place_pos_candidate = []
    self.even_place_pos_candidate = []

  def _getExistingXYPositions(self):
    positions = []
    for pos in self.odd_place_pos_candidate:
      positions.append(list(pos))
    for pos in self.even_place_pos_candidate:
      positions.append(list(pos))
    return positions

  def resetPallet(self):
    self.pallet_rz = np.random.random_sample() * np.pi if self.random_orientation else np.pi/2
    # self.pallet_rz = np.random.choice(np.linspace(0, np.pi, 8, endpoint=False))
    self.pallet_pos = self._getValidPositions(np.linalg.norm([self.pallet_size[0]/2, self.pallet_size[1]/2])*2, 0, [], 1)[0]
    self.pallet_pos.append(self.pallet_z)

    self.pallet = Pallet(self.pallet_pos, transformations.quaternion_from_euler(0, 0, self.pallet_rz),
                         np.random.choice(np.arange(self.block_scale_range[0], self.block_scale_range[1]+0.01, 0.02)))

    # pos candidate for odd layer
    dx = self.pallet_size[0] / 6
    dy = self.pallet_size[1] / 4
    pos_candidates = np.array([[2 * dx, -dy], [2 * dx, dy],
                               [0, -dy], [0, dy],
                               [-2 * dx, -dy], [-2 * dx, dy]])

    R = np.array([[np.cos(self.pallet_rz), -np.sin(self.pallet_rz)],
                  [np.sin(self.pallet_rz), np.cos(self.pallet_rz)]])
    transformed_pos_candidate = R.dot(pos_candidates.T).T
    self.even_place_pos_candidate = transformed_pos_candidate + self.pallet_pos[:2]

    # pos candidate for even layer
    dx = self.pallet_size[0] / 4
    dy = self.pallet_size[1] / 6
    pos_candidates = np.array([[dx, -2 * dy], [dx, 0], [dx, 2 * dy],
                               [-dx, -2 * dy], [-dx, 0], [-dx, 2 * dy]])

    R = np.array([[np.cos(self.pallet_rz), -np.sin(self.pallet_rz)],
                  [np.sin(self.pallet_rz), np.cos(self.pallet_rz)]])
    transformed_pos_candidate = R.dot(pos_candidates.T).T
    self.odd_place_pos_candidate = transformed_pos_candidate + self.pallet_pos[:2]

  def generateOneBox(self):
    while True:
      try:
        self._generateShapes(constants.BOX, 1, random_orientation=self.random_orientation)
      except NoValidPositionException:
        continue
      else:
        break
    self._changeBoxDynamics(self.objects[-1])

  def reset(self):
    while True:
      if self.pallet is not None:
        pb.removeBody(self.pallet.object_id)
      self.resetPybulletWorkspace()
      self.resetPallet()
      try:
        self._generateShapes(constants.BOX, 1, random_orientation=self.random_orientation)
        self._changeBoxDynamics(self.objects[-1])
      except NoValidPositionException:
        continue
      else:
        break
    return self._getObservation()

  def step(self, action):
    self.takeAction(action)
    self.wait(100)
    n_obj_on_ground = len(list(filter(lambda o: self._isObjOnGround(o), self.objects)))
    level1_objs, level2_objs, level3_objs = self.getObjEachLevel()
    # if level 2 is filled, freeze the level 1 boxes to speed up simulation
    if len(level2_objs) == 6 and len(level3_objs) == 0:
      for obj in level1_objs:
        pb.changeDynamics(obj.object_id, -1, mass=0)
        pb.resetBaseVelocity(obj.object_id, [0, 0, 0], [0, 0, 0])
    if self.isSimValid() and len(self.objects) < self.num_obj and not self._isHolding() and n_obj_on_ground == 0:
      self.generateOneBox()
    obs = self._getObservation(action)
    done = self._checkTermination()
    reward = 1.0 if done else 0.0

    if not done:
      done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
    self.current_episode_steps += 1

    return obs, reward, done

  def _checkTermination(self):
    n_level1, n_level2, n_level3 = self.getNEachLevel()
    all_upright = all(map(lambda o: self._checkObjUpright(o), self.objects))
    rz_valid = self.checkRzValid()
    if not (n_level1 + n_level2 + n_level3 == self.num_obj and all_upright and rz_valid):
      return False
    if self.num_obj <= 6:
      return True
    elif 6 < self.num_obj <= 12:
      return n_level1 == 6 and n_level2 == (self.num_obj - n_level1)
    elif 12 < self.num_obj <= 18:
      return n_level1 == 6 and n_level2 == 6 and n_level3 == (self.num_obj - n_level1 - n_level2)

  def getObjEachLevel(self):
    level1_threshold = self.pallet_height + 0.25 * self.box_height - 0.01
    level2_threshold = self.pallet_height + 1.25 * self.box_height - 0.01
    level3_threshold = self.pallet_height + 2.25 * self.box_height - 0.01
    level4_threshold = self.pallet_height + 3.25 * self.box_height - 0.01
    level1_objs = list(filter(lambda o: level1_threshold < o.getZPosition() < level2_threshold, self.objects))
    level2_objs = list(filter(lambda o: level2_threshold < o.getZPosition() < level3_threshold, self.objects))
    level3_objs = list(filter(lambda o: level3_threshold < o.getZPosition() < level4_threshold, self.objects))
    return level1_objs, level2_objs, level3_objs

  def getNEachLevel(self):
    level1_objs, level2_objs, level3_objs = self.getObjEachLevel()
    n_level1 = len(level1_objs)
    n_level2 = len(level2_objs)
    n_level3 = len(level3_objs)
    return n_level1, n_level2, n_level3

  def checkRzValid(self):
    level1_objs, level2_objs, level3_objs = self.getObjEachLevel()
    level1_rz = list(map(lambda o: transformations.euler_from_quaternion(o.getRotation())[2], level1_objs))
    level2_rz = list(map(lambda o: transformations.euler_from_quaternion(o.getRotation())[2], level2_objs))
    level3_rz = list(map(lambda o: transformations.euler_from_quaternion(o.getRotation())[2], level3_objs))
    level1_rz_goal = self.pallet_rz + np.pi / 2
    if level1_rz_goal > np.pi:
      level1_rz_goal -= np.pi
    level2_rz_goal = self.pallet_rz
    if level2_rz_goal > np.pi:
      level2_rz_goal -= np.pi
    level3_rz_goal = self.pallet_rz + np.pi / 2

    def rz_close(rz, goal):
      while rz < 0:
        rz += np.pi
      while rz > np.pi:
        rz -= np.pi
      angle_diff = abs(rz - goal)
      angle_diff = min(angle_diff, abs(angle_diff - np.pi))
      return angle_diff < np.pi/8

    level1_rz_ok = all(map(lambda rz: rz_close(rz, level1_rz_goal), level1_rz))
    level2_rz_ok = all(map(lambda rz: rz_close(rz, level2_rz_goal), level2_rz))
    level3_rz_ok = all(map(lambda rz: rz_close(rz, level3_rz_goal), level3_rz))
    return level1_rz_ok and level2_rz_ok and level3_rz_ok

  def _changeBoxDynamics(self, box):
    pb.changeDynamics(box.object_id, -1, linearDamping=0.04, angularDamping=0.04, restitution=0,
                      contactStiffness=3000, contactDamping=100)

def createBoxPalletizingEnv(config):
  return BoxPalletizingEnv(config)

if __name__ == '__main__':
  workspace = np.asarray([[0.2, 0.8],
                          [-0.3, 0.3],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 40, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyr', 'num_objects': 18, 'random_orientation': True,
                'reward_type': 'sparse', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point', 'physics_mode': 'fast', 'hard_reset_freq': 1000, 'object_scale_range': (1, 1),
                'kuka_adjust_gripper_offset': 0.001,
                }

  planner_config = {'random_orientation': True, 'half_rotation': True}

  env = BoxPalletizingEnv(env_config)
  planner = BoxPalletizingPlanner(env, planner_config)
  s, in_hand, obs = env.reset()
  while True:
    action = planner.getNextAction()
    obs, reward, done = env.step(action)

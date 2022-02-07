import pybullet as pb
import numpy as np

from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.envs.pybullet_envs.close_loop_envs.close_loop_env import CloseLoopEnv
from helping_hands_rl_envs.simulators.pybullet.utils import transformations
from helping_hands_rl_envs.planners.close_loop_household_pushing_planner import CloseLoopHouseholdPushingPlanner
from helping_hands_rl_envs.simulators.pybullet.equipments.tray import Tray
from helping_hands_rl_envs.simulators.constants import NoValidPositionException
from helping_hands_rl_envs.simulators.pybullet.utils.ortho_sensor import OrthographicSensor
from scipy.ndimage import rotate


class CloseLoopHouseholdPushingEnv(CloseLoopEnv):
  def __init__(self, config):
    super().__init__(config)
    self.object_init_z = 0.1
    if 'transparent_bin' not in config:
      self.trans_bin = False
    else:
      self.trans_bin = config['transparent_bin']
    if 'collision_penalty' not in config:
      self.coll_pen = True
    else:
      self.coll_pen = config['collision_penalty']
    self.fix_set = True
    if 'collision_terminate' not in config:
      self.collision_terminate = False
    else:
      self.collision_terminate = config['collision_terminate']
    self.tray = Tray()
    self.bin_size = 0.25


    cam_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 0.29]
    target_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 0]
    cam_up_vector = [-1, 0, 0]
    self.ws_size = max(self.workspace[0][1] - self.workspace[0][0], self.workspace[1][1] - self.workspace[1][0])
    self.sensor = OrthographicSensor(cam_pos, cam_up_vector, target_pos, self.ws_size, 0.1, 1)
    self.sensor.setCamMatrix(cam_pos, cam_up_vector, target_pos)

  def initialize(self):
    super().initialize()
    self.tray.initialize(pos=[self.workspace[0].mean(), self.workspace[1].mean(), 0],
                         size=[self.bin_size, self.bin_size, 0.1], transparent=self.trans_bin)

  def reset(self):
    self.resetPybulletEnv()
    self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
    while True:
      try:
        for i in range(self.num_obj):
          x = (np.random.rand() - 0.5) * 0.3
          x += self.workspace[0].mean()
          y = (np.random.rand() - 0.5) * 0.3
          y += self.workspace[1].mean()
          randpos = [x, y, 0.20]
          # obj = self._generateShapes(constants.RANDOM_HOUSEHOLD, 1, random_orientation=self.random_orientation,
          #                            pos=[randpos], padding=self.min_boarder_padding,
          #                            min_distance=self.min_object_distance, model_id=-1)
          obj = self._generateShapes(constants.RANDOM_HOUSEHOLD200, 1,
                                     random_orientation=self.random_orientation,
                                     pos=[randpos], padding=0.1,
                                     min_distance=0, model_id=i+3 if self.fix_set else -1)
          pb.changeDynamics(obj[0].object_id, -1, lateralFriction=0.6)
          self.wait(10)
      except NoValidPositionException:
        continue
      else:
        break
    self.wait(200)
    return self._getObservation()

  def _getValidOrientation(self, random_orientation):
    if random_orientation:
      orientation = pb.getQuaternionFromEuler([0., 0., np.pi/2 * (np.random.random_sample() - 0.5)])
    else:
      orientation = pb.getQuaternionFromEuler([0., 0., 0.])
    return orientation

  def _checkTermination(self):
    obj_bboxes = []
    for obj in self.objects:
      xyz_min, xyz_max = obj.getBoundingBox()
      obj_bboxes.append([xyz_min[0], xyz_min[1]])
      obj_bboxes.append([xyz_max[0], xyz_max[1]])
    obj_bboxes_min = np.stack(obj_bboxes).min(0)
    obj_bboxes_max = np.stack(obj_bboxes).max(0)
    return obj_bboxes_min[0] > self.workspace[0][0] + self.workspace_size/2 or \
           obj_bboxes_max[0] < self.workspace[0][1] - self.workspace_size/2 or \
           obj_bboxes_min[1] > self.workspace[1][0] + self.workspace_size/2 or \
           obj_bboxes_max[1] < self.workspace[1][1] - self.workspace_size/2


  def isSimValid(self):
    for obj in self.objects:
      p = obj.getPosition()
      if self._isObjectHeld(obj):
        continue
      if not self.workspace[0][0]-0.05 < p[0] < self.workspace[0][1]+0.05 and \
          self.workspace[1][0]-0.05 < p[1] < self.workspace[1][1]+0.05 and \
          self.workspace[2][0] < p[2] < self.workspace[2][1]:
        return False
    return True

  def _getHeightmap(self):
    gripper_pos = self.robot._getEndEffectorPosition()
    gripper_rz = transformations.euler_from_quaternion(self.robot._getEndEffectorRotation())[2]
    if self.view_type == 'render_center':
      return self.renderer.getTopDownDepth(self.workspace_size * 2, self.heightmap_size, gripper_pos, 0)
    elif self.view_type == 'render_center_height':
      depth = self.renderer.getTopDownDepth(self.workspace_size, self.heightmap_size, gripper_pos, 0)
      heightmap = gripper_pos[2] - depth
      return heightmap
    else:
      raise NotImplementedError

  def getGripperImg(self, gripper_state=None, gripper_rz=None):
    if gripper_state is None:
      gripper_state = self.robot.getGripperOpenRatio()
    if gripper_rz is None:
      gripper_rz = transformations.euler_from_quaternion(self.robot._getEndEffectorRotation())[2]
    im = np.zeros((self.heightmap_size, self.heightmap_size))
    if self.robot_type == 'panda':
      d = int(42/2/128*self.heightmap_size * gripper_state)
    elif self.robot_type == 'kuka':
      d = int(45/2/128*self.heightmap_size * gripper_state)
    else:
      raise NotImplementedError
    anchor = self.heightmap_size//2
    im[anchor - d // 2 - 3:anchor - d // 2 + 3, anchor - 3:anchor + 3] = 1
    im[anchor + d // 2 - 3:anchor + d // 2 + 3, anchor - 3:anchor + 3] = 1
    im = rotate(im, np.rad2deg(gripper_rz), reshape=False, order=0)
    return im

  # def _getObservation(self, action=None):
  #   self.heightmap = self._getHeightmap()
  #   gripper_img = self.getGripperImg()
  #   heightmap = self.heightmap
  #   heightmap[gripper_img == 1] = 0
  #   heightmap = heightmap.reshape([1, self.heightmap_size, self.heightmap_size])
  #
  #   bin_img = self.renderer.getTopDownDepth(self.workspace_size, self.heightmap_size, [self.workspace[0].mean(), self.workspace[1].mean(), self.workspace[2][1]], 0)
  #   bin_img = bin_img.reshape([1, self.heightmap_size, self.heightmap_size])
  #   obs = np.concatenate([heightmap, bin_img])
  #   return self._isHolding(), None, obs

def createCloseLoopHouseholdPushingEnv(config):
  return CloseLoopHouseholdPushingEnv(config)

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  workspace = np.asarray([[0.3, 0.6],
                          [-0.15, 0.15],
                          [0.01, 0.25]])
  env_config = {'workspace': workspace, 'max_steps': 20, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyzr', 'num_objects': 4, 'random_orientation': True,
                'reward_type': 'sparse', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point', 'physics_mode': 'fast', 'hard_reset_freq': 1000, 'object_scale_range': (0.8, 1),
                'view_type': 'render_center', 'transparent_bin': False, 'collision_penalty': False}

  planner_config = {'random_orientation': True, 'dpos': 0.05, 'drot': np.pi/8}

  env_config['seed'] = 1
  env = CloseLoopHouseholdPushingEnv(env_config)
  planner = CloseLoopHouseholdPushingPlanner(env, planner_config)
  s, in_hand, obs = env.reset()
  # while True:
  #   current_pos = env.robot._getEndEffectorPosition()
  #   current_rot = transformations.euler_from_quaternion(env.robot._getEndEffectorRotation())
  #
  #   block_pos = env.objects[0].getPosition()
  #   block_rot = transformations.euler_from_quaternion(env.objects[0].getRotation())
  #
  #   pos_diff = block_pos - current_pos
  #   rot_diff = np.array(block_rot) - current_rot
  #   pos_diff[pos_diff // 0.01 > 1] = 0.01
  #   pos_diff[pos_diff // -0.01 > 1] = -0.01
  #
  #   rot_diff[rot_diff // (np.pi/32) > 1] = np.pi/32
  #   rot_diff[rot_diff // (-np.pi/32) > 1] = -np.pi/32
  #
  #   action = [1, pos_diff[0], pos_diff[1], pos_diff[2], rot_diff[2]]
  #   obs, reward, done = env.step(action)

  while True:
    action = planner.getNextAction()
    obs, reward, done = env.step(action)
    if reward == 1:
      print(1)
    if done == 1:
      print(2)

  # fig, axs = plt.subplots(8, 5, figsize=(25, 40))
  # for i in range(40):
  #   action = planner.getNextAction()
  #   obs, reward, done = env.step(action)
  #   axs[i//5, i%5].imshow(obs[2][0], vmax=0.3)
  # env.reset()
  # fig.show()
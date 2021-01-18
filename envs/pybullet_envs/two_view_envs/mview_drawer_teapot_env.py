import pybullet as pb
import numpy as np
import os
import helping_hands_rl_envs

import matplotlib.pyplot as plt

from helping_hands_rl_envs.envs.pybullet_envs.two_view_envs.drawer_teapot_env import DrawerTeapotEnv
from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators.pybullet.utils.sensor import Sensor
from helping_hands_rl_envs.simulators.pybullet.utils.renderer import Renderer

class MViewDrawerTeapotEnv(DrawerTeapotEnv):
  def __init__(self, config):
    super().__init__(config)
    self.wall_x = 1.1
    cam_forward_target_pos = [0.9, self.workspace[1].mean(), self.workspace[2].mean()]
    cam_forward_up_vector = [0, 0, 1]

    cam_1_forward_pos = [0, 0, self.workspace[2][1]]
    far_1 = np.linalg.norm(np.array(cam_1_forward_pos) - np.array(cam_forward_target_pos)) + 2
    self.sensor_1 = Sensor(cam_1_forward_pos, cam_forward_up_vector, cam_forward_target_pos,
                           4 * (self.workspace[2][1] - self.workspace[2][0]), near=0.1, far=far_1)

    cam_2_forward_pos = [0, 0.5, self.workspace[2][1]]
    far_2 = np.linalg.norm(np.array(cam_2_forward_pos) - np.array(cam_forward_target_pos)) + 2
    self.sensor_2 = Sensor(cam_2_forward_pos, cam_forward_up_vector, cam_forward_target_pos,
                           4*(self.workspace[2][1] - self.workspace[2][0]), near=0.1, far=far_2)

    cam_3_forward_pos = [0, -0.5, self.workspace[2][1]]
    far_3 = np.linalg.norm(np.array(cam_3_forward_pos) - np.array(cam_forward_target_pos)) + 2
    self.sensor_3 = Sensor(cam_3_forward_pos, cam_forward_up_vector, cam_forward_target_pos,
                           4*(self.workspace[2][1] - self.workspace[2][0]), near=0.1, far=far_3)

    self.renderer = Renderer()

    self.wall_id = None

  def initialize(self):
    super().initialize()
    root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
    urdf_filepath = os.path.join(root_dir, 'simulators/urdf/', 'wall.urdf')
    self.wall_id = pb.loadURDF(urdf_filepath,
                                     [self.wall_x,
                                      self.workspace[1].mean(),
                                      0],
                                     pb.getQuaternionFromEuler([0, 0, 0]),
                                     globalScaling=1)

  def _getHeightmapForward(self):
    return self.sensor_2.getHeightmap(self.heightmap_size)

  def getHeightmapByThetas(self, thetas):
    points1 = self.sensor_1.getPointCloud(self.heightmap_size * 2)
    points2 = self.sensor_2.getPointCloud(self.heightmap_size * 2)
    points3 = self.sensor_3.getPointCloud(self.heightmap_size * 2)

    self.renderer.clearPoints()
    self.renderer.addPoints(points1)
    self.renderer.addPoints(points2)
    self.renderer.addPoints(points3)

    heightmaps = []
    for theta in thetas:
      dy = np.sin(theta) * 1
      dx = np.cos(theta) * 1

      render_cam_target_pos = [self.workspace[0][1], self.workspace[1].mean(), self.workspace[2].mean()]
      render_cam_up_vector = [0, 0, 1]

      render_cam_pos1 = [self.wall_x-dx, dy, self.workspace[2].mean()]
      far = np.linalg.norm(np.array(render_cam_pos1) - np.array(render_cam_target_pos))+1
      # far = 14
      hm = self.renderer.renderHeightmapOrthographic(self.heightmap_size, render_cam_pos1, render_cam_up_vector,
                                          render_cam_target_pos, self.workspace[2][1] - self.workspace[2][0], 0.1, far)
      heightmaps.append(hm)
      plt.imshow(hm)
      plt.show()
    heightmaps = np.stack(heightmaps)
    return heightmaps

  def _getObservation(self, action=None):
    ''''''
    # TODO:
    # heightmaps = self.getHeightmapByThetas([-np.pi/6, 0, np.pi/6])
    heightmaps = self.getHeightmapByThetas([0])
    old_heightmap = self.heightmap
    self.heightmap = self._getHeightmap()

    if action is None or self._isHolding() == False:
      in_hand_img = self.getEmptyInHand()
    else:
      motion_primative, x, y, z, rot = self._decodeAction(action)
      in_hand_img = self.getInHandImage(old_heightmap, x, y, z, rot, self.heightmap)

    forward_heightmap = self._getHeightmapForward()
    heightmaps = np.stack((self.heightmap, forward_heightmap), 0)
    heightmaps = np.moveaxis(heightmaps, 0, -1)

    return self._isHolding(), in_hand_img, heightmaps

if __name__ == '__main__':
  workspace = np.asarray([[0.3, 0.7],
                          [-0.2, 0.2],
                          [0, 0.40]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyrr', 'num_objects': 5, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point', 'physics_mode': 'slow'}
  env = MViewDrawerTeapotEnv(env_config)
  while True:
    s, in_hand, obs = env.reset()

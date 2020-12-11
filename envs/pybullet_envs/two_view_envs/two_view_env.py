import pybullet as pb
import numpy as np
import os
import helping_hands_rl_envs

from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators.pybullet.utils.sensor import Sensor


class TwoViewEnv(PyBulletEnv):
  def __init__(self, config):
    super().__init__(config)
    self.wall_x = 1.1

    cam_forward_pos = [-10, self.workspace[1].mean(), self.workspace[2].mean()]
    cam_forward_target_pos = [self.wall_x, self.workspace[1].mean(), self.workspace[2].mean()]
    cam_forward_up_vector = [0, 0, 1]
    self.sensor_forward = Sensor(cam_forward_pos, cam_forward_up_vector, cam_forward_target_pos,
                                 self.workspace[2][1] - self.workspace[2][0], near=10, far=self.wall_x+10)

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
    return self.sensor_forward.getHeightmap(self.heightmap_size)

  def _getObservation(self, action=None):
    ''''''
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


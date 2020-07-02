from copy import deepcopy
import pybullet as pb
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv, NoValidPositionException
from helping_hands_rl_envs.simulators.pybullet.robots.kuka_float_pick import KukaFloatPick
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.utils import pybullet_util
import numpy.random as npr
import numpy as np
import sys
import os
import helping_hands_rl_envs


class PyBulletTiltEnv(PyBulletEnv):
  def __init__(self, config):
    super().__init__(config)

    self.rx_range = (0, np.pi/6)
    self.tilt_plain_rx = 0
    self.tilt_plain2_rx = 0
    self.tilt_plain_id = -1
    self.tilt_plain2_id = -1
    self.pick_rx = 0
    self.tilt_border = 0.035
    self.tilt_border2 = 0.035
    self.tilt_z1 = 0
    self.tilt_z2 = 0

    self.tilt_rz = 0

    self.block_scale_range = (0.6, 0.6)
    self.min_block_size = self.block_original_size * self.block_scale_range[0]
    self.max_block_size = self.block_original_size * self.block_scale_range[1]

  def initialize(self):
    super().initialize()
    self.tilt_plain_id = -1
    self.tilt_plain2_id = -1

  def resetTilt(self):
    self.tilt_rz = -np.pi / 2 + np.random.random_sample() * np.pi
    self.tilt_border = np.random.random() * 0.08 + 0.02
    self.tilt_border2 = np.random.random() * 0.08 + 0.02
    self.tilt_z1 = np.random.random()*0.01
    self.tilt_z2 = np.random.random()*0.01

    if self.tilt_plain_id > -1:
      pb.removeBody(self.tilt_plain_id)
    if self.tilt_plain2_id > -1:
      pb.removeBody(self.tilt_plain2_id)
    root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
    urdf_filepath = os.path.join(root_dir, constants.URDF_PATH, 'tilt.urdf')

    self.tilt_plain_rx = (self.rx_range[1] - self.rx_range[0]) * np.random.random_sample() + self.rx_range[0]
    self.tilt_plain_id = pb.loadURDF(urdf_filepath,
                                     [self.workspace[0].mean() - self.tilt_border * np.sin(self.tilt_rz),
                                      self.tilt_border * np.cos(self.tilt_rz),
                                      self.tilt_z1],
                                     pb.getQuaternionFromEuler([self.tilt_plain_rx, 0, self.tilt_rz]),
                                     globalScaling=1)
    self.tilt_plain2_rx = (self.rx_range[0] - self.rx_range[1]) * np.random.random_sample() + self.rx_range[0]
    self.tilt_plain2_id = pb.loadURDF(urdf_filepath,
                                      [self.workspace[0].mean() + self.tilt_border2 * np.sin(self.tilt_rz),
                                       -self.tilt_border2 * np.cos(self.tilt_rz),
                                       self.tilt_z2],
                                      pb.getQuaternionFromEuler([-self.tilt_plain2_rx, 0, self.tilt_rz+np.pi]),
                                      globalScaling=1)
  
  def getY1Y2fromX(self, x):
    y1 = np.tan(self.tilt_rz) * x - np.tan(self.tilt_rz) * (self.workspace[0].mean() - self.tilt_border/np.sin(self.tilt_rz))
    y2 = np.tan(self.tilt_rz) * x - np.tan(self.tilt_rz) * (self.workspace[0].mean() + self.tilt_border2/np.sin(self.tilt_rz))
    return y1, y2    
  
  def isPosOffTilt(self, pos):
    y1, y2 = self.getY1Y2fromX(pos[0])
    return y2+0.02/np.cos(self.tilt_rz) < pos[1] < y1-0.02/np.cos(self.tilt_rz)

  def isPosDistToTiltValid(self, pos, obj_type):
    y1, y2 = self.getY1Y2fromX(pos[0])
    d1 = abs(pos[1] - y1) * abs(np.cos(self.tilt_rz))
    d2 = abs(y2 - pos[1]) * abs(np.cos(self.tilt_rz))
    if obj_type in (constants.BRICK, constants.ROOF):
      d_threshold = self.max_block_size * 1.6
    else:
      d_threshold = self.max_block_size * 0.7
    return d1 > d_threshold and d2 > d_threshold

  def resetWithTiltAndObj(self, obj_dict):
    while True:
      super().reset()
      self.resetTilt()
      try:
        existing_pos = []
        for t in obj_dict:
          padding = pybullet_util.getPadding(t, self.max_block_size)
          min_distance = pybullet_util.getMinDistance(t, self.max_block_size)
          for j in range(100):
            if t == constants.CUBE:
              for i in range(100):
                off_tilt_pos = self._getValidPositions(padding, min_distance, existing_pos, 1)
                if self.isPosOffTilt(off_tilt_pos[0]):
                  break
              if i == 100:
                raise NoValidPositionException
              other_pos = self._getValidPositions(padding, min_distance, existing_pos+off_tilt_pos, obj_dict[t]-1)
              other_pos.extend(off_tilt_pos)
            else:
              other_pos = self._getValidPositions(padding, min_distance, existing_pos, obj_dict[t])
            if all(map(lambda p: self.isPosDistToTiltValid(p, t), other_pos)):
              break

          orientations = []
          existing_pos.extend(deepcopy(other_pos))
          for position in other_pos:
            y1, y2 = self.getY1Y2fromX(position[0])
            if position[1] > y1:
              d = (position[1] - y1) * np.cos(self.tilt_rz)
              position.append(self.tilt_z1 + 0.02+np.tan(self.tilt_plain_rx) * d)
              orientations.append(pb.getQuaternionFromEuler([self.tilt_plain_rx, 0, self.tilt_rz]))
            elif position[1] < y2:
              d = (y2 - position[1]) * np.cos(self.tilt_rz)
              position.append(self.tilt_z2 + 0.02+np.tan(-self.tilt_plain2_rx) * d)
              orientations.append(pb.getQuaternionFromEuler([self.tilt_plain2_rx, 0, self.tilt_rz]))
            else:
              position.append(0.02)
              orientations.append(pb.getQuaternionFromEuler([0, 0, np.random.random()*np.pi*2]))
          self._generateShapes(t, obj_dict[t], random_orientation=False, pos=other_pos, rot=orientations)
      except Exception as e:
        continue
      else:
        break




from copy import deepcopy
import pybullet as pb
from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils.constants import NoValidPositionException
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils import pybullet_util
from bulletarm.pybullet.utils import transformations
import numpy.random as npr
import numpy as np
import sys
import os
import bulletarm


class RampBaseEnv(BaseEnv):
  def __init__(self, config):
    super().__init__(config)

    self.rx_range = (0, np.pi/6)
    self.ramp1_angle = 0
    self.ramp2_angle = 0
    self.ramp1_id = -1
    self.ramp2_id = -1
    self.ramp1_dist_to_center = 0.035
    self.ramp2_dist_to_center = 0.035
    self.ramp1_height = 0
    self.ramp2_height = 0

    self.min_ramp_dist = 0.03
    self.max_ramp_dist = 0.1

    self.ramp_rz = 0

  def initialize(self):
    super().initialize()
    self.ramp1_id = -1
    self.ramp2_id = -1

  def resetRamp(self):
    self.ramp_rz = -np.pi / 2 + np.random.random_sample() * np.pi
    self.ramp1_dist_to_center = np.random.random() * (self.max_ramp_dist - self.min_ramp_dist) + self.min_ramp_dist
    self.ramp2_dist_to_center = np.random.random() * (self.max_ramp_dist - self.min_ramp_dist) + self.min_ramp_dist
    self.ramp1_height = np.random.random() * 0.01
    self.ramp2_height = np.random.random() * 0.01

    if self.ramp1_id > -1:
      pb.removeBody(self.ramp1_id)
    if self.ramp2_id > -1:
      pb.removeBody(self.ramp2_id)
    root_dir = os.path.dirname(bulletarm.__file__)
    urdf_filepath = os.path.join(root_dir, constants.OBJECTS_PATH, 'tilt.urdf')

    self.ramp1_angle = (self.rx_range[1] - self.rx_range[0]) * np.random.random_sample() + self.rx_range[0]
    self.ramp1_id = pb.loadURDF(urdf_filepath,
                                [self.workspace[0].mean() - self.ramp1_dist_to_center * np.sin(self.ramp_rz),
                                 self.ramp1_dist_to_center * np.cos(self.ramp_rz),
                                 self.ramp1_height],
                                pb.getQuaternionFromEuler([self.ramp1_angle, 0, self.ramp_rz]),
                                globalScaling=1)
    self.ramp2_angle = (self.rx_range[0] - self.rx_range[1]) * np.random.random_sample() + self.rx_range[0]
    self.ramp2_id = pb.loadURDF(urdf_filepath,
                                [self.workspace[0].mean() + self.ramp2_dist_to_center * np.sin(self.ramp_rz),
                                 -self.ramp2_dist_to_center * np.cos(self.ramp_rz),
                                 self.ramp2_height],
                                pb.getQuaternionFromEuler([-self.ramp2_angle, 0, self.ramp_rz + np.pi]),
                                globalScaling=1)
    pb.changeVisualShape(self.ramp1_id, -1, rgbaColor=[0.8706, 0.7216, 0.5294, 1])
    pb.changeVisualShape(self.ramp2_id, -1, rgbaColor=[0.8706, 0.7216, 0.5294, 1])

  def getY1Y2fromX(self, x):
    y1 = np.tan(self.ramp_rz) * x - np.tan(self.ramp_rz) * (self.workspace[0].mean() - self.ramp1_dist_to_center / np.sin(self.ramp_rz))
    y2 = np.tan(self.ramp_rz) * x - np.tan(self.ramp_rz) * (self.workspace[0].mean() + self.ramp2_dist_to_center / np.sin(self.ramp_rz))
    return y1, y2

  def isPosOffRamp(self, pos, min_dist=0.02):
    y1, y2 = self.getY1Y2fromX(pos[0])
    return y2 + min_dist / np.cos(self.ramp_rz) < pos[1] < y1 - min_dist / np.cos(self.ramp_rz)

  def isPosDistToRampValid(self, pos, obj_type):
    y1, y2 = self.getY1Y2fromX(pos[0])
    d1 = abs(pos[1] - y1) * abs(np.cos(self.ramp_rz))
    d2 = abs(y2 - pos[1]) * abs(np.cos(self.ramp_rz))
    if obj_type in (constants.BRICK, constants.ROOF):
      d_threshold = self.max_block_size * 1.6
    else:
      d_threshold = self.max_block_size * 0.7
    return d1 > d_threshold and d2 > d_threshold

  def resetWithRampAndObj(self, obj_dict):
    while True:
      self.resetPybulletWorkspace()
      self.resetRamp()
      try:
        existing_pos = []
        for t in obj_dict:
          padding = pybullet_util.getPadding(t, self.max_block_size)
          min_distance = pybullet_util.getMinDistance(t, self.max_block_size)
          for j in range(100):
            # put at least one cube or random shape on the ground
            if t == constants.CUBE or t == constants.RANDOM:
              for i in range(100):
                off_tilt_pos = self._getValidPositions(padding, min_distance, existing_pos, 1)
                if self.isPosOffRamp(off_tilt_pos[0]):
                  break
              if i == 100:
                raise NoValidPositionException
              positions = self._getValidPositions(padding, min_distance, existing_pos+off_tilt_pos, obj_dict[t]-1)
              positions.extend(off_tilt_pos)
            else:
              positions = self._getValidPositions(padding, min_distance, existing_pos, obj_dict[t])
            if all(map(lambda p: self.isPosDistToRampValid(p, t), positions)):
              break

          existing_pos.extend(deepcopy(positions))
          orientations = self.calculateOrientations(positions)
          if t == constants.RANDOM:
            self.generateRandomShape(obj_dict[t], positions, orientations)
          elif t == constants.BRICK:
            self.generateBrickShape(obj_dict[t], positions, orientations)
          else:
            self._generateShapes(t, obj_dict[t], random_orientation=False, pos=positions, rot=orientations)
      except Exception as e:
        continue
      else:
        break

  def generateRandomShape(self, n, poss, rots):
    self._generateShapes(constants.RANDOM, n, random_orientation=False, pos=poss, rot=rots)

  def generateBrickShape(self, n, poss, rots):
    self._generateShapes(constants.BRICK, n, random_orientation=False, pos=poss, rot=rots)

  def calculateOrientations(self, positions):
    orientations = []
    for position in positions:
      y1, y2 = self.getY1Y2fromX(position[0])
      if position[1] > y1:
        d = (position[1] - y1) * np.cos(self.ramp_rz)
        position.append(self.ramp1_height + 0.02 + np.tan(self.ramp1_angle) * d)
        rx = self.ramp1_angle
        rz = self.ramp_rz
        # orientations.append(pb.getQuaternionFromEuler([self.tilt_plain_rx, 0, self.tilt_rz]))
      elif position[1] < y2:
        d = (y2 - position[1]) * np.cos(self.ramp_rz)
        position.append(self.ramp2_height + 0.02 + np.tan(-self.ramp2_angle) * d)
        rx = self.ramp2_angle
        rz = self.ramp_rz
        # orientations.append(pb.getQuaternionFromEuler([self.tilt_plain2_rx, 0, self.tilt_rz]))
      else:
        position.append(0.02)
        # orientations.append(pb.getQuaternionFromEuler([0, 0, np.random.random()*np.pi*2]))
        rx = 0
        rz = np.random.random() * np.pi * 2
      T = transformations.euler_matrix(rz, 0, rx)
      T_random = transformations.euler_matrix(np.random.random() * np.pi, 0, 0)
      T = T_random.dot(T)
      rz, ry, rx = transformations.euler_from_matrix(T)
      orientations.append(pb.getQuaternionFromEuler([rx, ry, rz]))
    return orientations


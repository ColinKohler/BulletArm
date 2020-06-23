from copy import deepcopy
import pybullet as pb
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators.pybullet.robots.kuka_float_pick import KukaFloatPick
from helping_hands_rl_envs.simulators import constants
import numpy.random as npr
import numpy as np

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
    self.tilt_border2 = -0.035

    self.tilt_rz = 0

  def initialize(self):
    super().initialize()
    self.tilt_plain_id = -1
    self.tilt_plain2_id = -1

  def resetTilt(self):
    self.tilt_rz = -np.pi / 2 + np.random.random_sample() * np.pi
    if self.tilt_plain_id > -1:
      pb.removeBody(self.tilt_plain_id)
    if self.tilt_plain2_id > -1:
      pb.removeBody(self.tilt_plain2_id)

    self.tilt_plain_rx = (self.rx_range[1] - self.rx_range[0]) * np.random.random_sample() + self.rx_range[0]
    self.tilt_plain_id = pb.loadURDF('plane.urdf',
                                     [self.workspace[0].mean() - self.tilt_border * np.sin(self.tilt_rz),
                                      self.tilt_border * np.cos(self.tilt_rz),
                                      0],
                                     pb.getQuaternionFromEuler([self.tilt_plain_rx, 0, self.tilt_rz]),
                                     globalScaling=0.002)
    self.tilt_plain2_rx = (self.rx_range[0] - self.rx_range[1]) * np.random.random_sample() + self.rx_range[0]
    self.tilt_plain2_id = pb.loadURDF('plane.urdf',
                                      [self.workspace[0].mean() + self.tilt_border * np.sin(self.tilt_rz),
                                       -self.tilt_border * np.cos(self.tilt_rz),
                                       0],
                                      pb.getQuaternionFromEuler([self.tilt_plain2_rx, 0, self.tilt_rz]),
                                      globalScaling=0.002)

  def resetWithTiltAndObj(self, obj_dict):
    while True:
      self.resetTilt()
      try:
        existing_pos = []
        for t in obj_dict:
          if t in (constants.CUBE, constants.TRIANGLE, constants.RANDOM):
            padding = self.max_block_size * 1.5
            min_distance = self.max_block_size * 2.4
          elif t in (constants.BRICK, constants.ROOF):
            padding = self.max_block_size * 3.4
            min_distance = self.max_block_size * 3.4
          else:
            padding = self.max_block_size * 1.5
            min_distance = self.max_block_size * 2.4

          other_pos = self._getValidPositions(padding, min_distance, existing_pos, obj_dict[t])
          orientations = []
          existing_pos.extend(deepcopy(other_pos))
          for position in other_pos:
            y1 = np.tan(self.tilt_rz) * position[0] - (self.workspace[0].mean()*np.tan(self.tilt_rz) - self.tilt_border)
            y2 = np.tan(self.tilt_rz) * position[0] - (self.workspace[0].mean()*np.tan(self.tilt_rz) + self.tilt_border)
            if position[1] > y1:
              d = (position[1] - y1) * np.cos(self.tilt_rz)
              position.append(0.02+np.tan(self.tilt_plain_rx) * d)
              orientations.append(pb.getQuaternionFromEuler([self.tilt_plain_rx, 0, self.tilt_rz]))
            elif position[1] < y2:
              d = (y2 - position[1]) * np.cos(self.tilt_rz)
              position.append(0.02+np.tan(-self.tilt_plain2_rx) * d)
              orientations.append(pb.getQuaternionFromEuler([self.tilt_plain2_rx, 0, self.tilt_rz]))
            else:
              position.append(0.02)
              orientations.append(pb.getQuaternionFromEuler([0, 0, np.random.random()*np.pi*2]))
          self._generateShapes(t, obj_dict[t], random_orientation=False, pos=other_pos, rot=orientations)
      except Exception as e:
        continue
      else:
        break



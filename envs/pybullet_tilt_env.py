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

  def resetWithTiltAndObj(self, obj_dict):
    while True:
      if self.tilt_plain_id > -1:
        pb.removeBody(self.tilt_plain_id)
      if self.tilt_plain2_id > -1:
        pb.removeBody(self.tilt_plain2_id)

      super(PyBulletTiltEnv, self).reset()
      self.tilt_plain_rx = (self.rx_range[1] - self.rx_range[0]) * np.random.random_sample() + self.rx_range[0]
      self.tilt_plain_id = pb.loadURDF('plane.urdf', [0.5 * (self.workspace[0][1] + self.workspace[0][0]), self.tilt_border, 0], pb.getQuaternionFromEuler([self.tilt_plain_rx, 0, 0]),
                                       globalScaling=0.005)
      self.tilt_plain2_rx = (self.rx_range[0] - self.rx_range[1]) * np.random.random_sample() + self.rx_range[0]
      self.tilt_plain2_id = pb.loadURDF('plane.urdf',
                                       [0.5 * (self.workspace[0][1] + self.workspace[0][0]), self.tilt_border2, 0],
                                       pb.getQuaternionFromEuler([self.tilt_plain2_rx, 0, 0]),
                                       globalScaling=0.005)
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
            if position[1] < self.tilt_border2:
              position.append(0.01 + np.tan(-self.tilt_plain2_rx) * -position[1])
              orientations.append(pb.getQuaternionFromEuler([self.tilt_plain2_rx, 0, 0]))
            elif position[1] > self.tilt_border:
              position.append(0.01 + np.tan(self.tilt_plain_rx) * position[1])
              orientations.append(pb.getQuaternionFromEuler([self.tilt_plain_rx, 0, 0]))
            else:
              position.append(0.02)
              orientations.append(pb.getQuaternionFromEuler([0, 0, np.random.random()*np.pi*2]))
          self._generateShapes(t, obj_dict[t], random_orientation=False, pos=other_pos, rot=orientations)

        # existing_pos = []
        # count = 0
        # for t in obj_dict:
        #   for i in range(obj_dict[t]):
        #     if np.random.random() > 0.5:
        #       other_pos = self._getValidPositions(self.max_block_size * 3, self.max_block_size * 3, existing_pos, 1,
        #                                           sample_range=[self.workspace[0],
        #                                                         [self.tilt_border + 0.02, self.workspace[1][1]]])
        #       existing_pos.extend(deepcopy(other_pos))
        #       for position in other_pos:
        #         position.append(0.04 + np.tan(self.tilt_plain_rx) * position[1])
        #       orientations = [pb.getQuaternionFromEuler([self.tilt_plain_rx, 0, 0])]
        #
        #     else:
        #       other_pos = self._getValidPositions(self.max_block_size * 3, self.max_block_size * 3, existing_pos, 1,
        #                                           sample_range=[self.workspace[0],
        #                                                         [self.workspace[1][0], self.tilt_border2 - 0.02]])
        #       existing_pos.extend(deepcopy(other_pos))
        #       for position in other_pos:
        #         position.append(0.04 + np.tan(-self.tilt_plain2_rx) * -position[1])
        #       orientations = [pb.getQuaternionFromEuler([self.tilt_plain2_rx, 0, 0])]
        #     self._generateShapes(t, 1, random_orientation=False, pos=other_pos, rot=orientations)
        #     count += 1
      except Exception as e:
        continue
      else:
        break



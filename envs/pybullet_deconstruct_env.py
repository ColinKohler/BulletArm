import pybullet as pb
import numpy as np
import scipy
import numpy.random as npr

from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv, NoValidPositionException
import helping_hands_rl_envs.simulators.pybullet.utils.object_generation as pb_obj_generation
from helping_hands_rl_envs.simulators import constants

class PyBulletDeconstructEnv(PyBulletEnv):
  def __init__(self, config):
    super().__init__(config)
    self.pick_offset = -0.007
    self.structure_objs = []

  def _getObservation(self, action=None):
    ''''''
    old_heightmap = self.heightmap

    image_arr = pb.getCameraImage(width=self.heightmap_size, height=self.heightmap_size,
                                  viewMatrix=self.view_matrix, projectionMatrix=self.proj_matrix)
    self.heightmap = image_arr[3] - np.min(image_arr[3])

    if action is None or self._isHolding() == True:
      in_hand_img = np.zeros((self.in_hand_size, self.in_hand_size, 1))
    else:
      motion_primative, x, y, z, rot = self._decodeAction(action)
      in_hand_img = self.getInHandImage(self.heightmap, x, y, rot, old_heightmap)


    return self._isHolding(), in_hand_img, self.heightmap.reshape([self.heightmap_size, self.heightmap_size, 1])

  def reset(self):
    super().reset()
    self.structure_objs = []

  def generateImproviseH3(self):
    lower_z1 = 0.01
    lower_z2 = 0.025
    hier_z = 0.02
    roof_z = 0.05

    def generate(pos, zscale=1):
      handle = pb_obj_generation.generateRandomObj(pos,
                                                   pb.getQuaternionFromEuler(
                                                     [0., 0., 2 * np.pi * np.random.random_sample()]),
                                                   npr.uniform(self.block_scale_range[0], self.block_scale_range[1]),
                                                   zscale)
      self.objects.append(handle)
      self.object_types[handle] = constants.RANDOM
      self.structure_objs.append(handle)

    padding = self.max_block_size * 1.5
    pos1 = self._getValidPositions(padding, 0, [], 1)[0]
    min_dist = 1.7 * self.max_block_size
    max_dist = 2.4 * self.max_block_size
    sample_range = [[pos1[0] - max_dist, pos1[0] + max_dist],
                    [pos1[1] - max_dist, pos1[1] + max_dist]]
    for i in range(100):
      try:
        pos2 = self._getValidPositions(padding, min_dist, [pos1], 1, sample_range=sample_range)[0]
      except NoValidPositionException:
        continue
      dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
      if min_dist < dist < max_dist:
        break

    t = np.random.choice(4)
    if t == 0:
      generate([pos1[0], pos1[1], lower_z1], 1)
      generate([pos1[0], pos1[1], lower_z2], 1)
      generate([pos2[0], pos2[1], lower_z1], 1)
      generate([pos2[0], pos2[1], lower_z2], 1)

    elif t == 1:
      generate([pos1[0], pos1[1], lower_z1], 1)
      generate([pos1[0], pos1[1], lower_z2], 1)
      generate([pos2[0], pos2[1], hier_z], 2)

      self._generateShapes(constants.RANDOM, 1, random_orientation=True, z_scale=np.random.choice([1, 2]))

    elif t == 2:
      generate([pos1[0], pos1[1], hier_z], 2)
      generate([pos2[0], pos2[1], lower_z1], 1)
      generate([pos2[0], pos2[1], lower_z2], 1)

      self._generateShapes(constants.RANDOM, 1, random_orientation=True, z_scale=np.random.choice([1, 2]))

    elif t == 3:
      generate([pos1[0], pos1[1], hier_z], 2)
      generate([pos2[0], pos2[1], hier_z], 2)

      self._generateShapes(constants.RANDOM, 2, random_orientation=True, z_scale=np.random.choice([1, 2]))

    obj_positions = np.array([pos1, pos2])
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(obj_positions[:, 0], obj_positions[:, 1])
    x, y = obj_positions.mean(0)
    r = np.arctan(slope)
    while r > np.pi:
      r -= np.pi

    h = pb_obj_generation.generateRoof([x, y, roof_z],
                                       pb.getQuaternionFromEuler(
                                         [0., 0., r]),
                                       npr.uniform(self.block_scale_range[0], self.block_scale_range[1]))
    self.objects.append(h)
    self.object_types[h] = constants.ROOF
    self.structure_objs.append(h)
    self.wait(50)

  def generateImproviseH4(self):
    roof_z = 0.06

    def generate(pos, zscale=1):
      handle = pb_obj_generation.generateRandomObj(pos,
                                                   pb.getQuaternionFromEuler(
                                                     [0., 0., 2 * np.pi * np.random.random_sample()]),
                                                   npr.uniform(self.block_scale_range[0], self.block_scale_range[1]),
                                                   zscale)
      self.objects.append(handle)
      self.object_types[handle] = constants.RANDOM
      self.structure_objs.append(handle)

    padding = self.max_block_size * 1.5
    pos1 = self._getValidPositions(padding, 0, [], 1)[0]
    min_dist = 1.7 * self.max_block_size
    max_dist = 2.4 * self.max_block_size
    sample_range = [[pos1[0] - max_dist, pos1[0] + max_dist],
                    [pos1[1] - max_dist, pos1[1] + max_dist]]
    for i in range(100):
      try:
        pos2 = self._getValidPositions(padding, min_dist, [pos1], 1, sample_range=sample_range)[0]
      except NoValidPositionException:
        continue
      dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
      if min_dist < dist < max_dist:
        break

    base1_scale1 = np.random.uniform(1, 2)
    base1_scale2 = 3 - base1_scale1

    base2_scale1 = np.random.uniform(1, 2)
    base2_scale2 = 3 - base2_scale1

    generate([pos1[0], pos1[1], base1_scale1 * 0.007], base1_scale1)
    generate([pos1[0], pos1[1], base1_scale1 * 0.014 + base1_scale2 * 0.007], base1_scale2)

    generate([pos2[0], pos2[1], base2_scale1 * 0.007], base2_scale1)
    generate([pos2[0], pos2[1], base2_scale1 * 0.014 + base2_scale2 * 0.007], base2_scale2)

    obj_positions = np.array([pos1, pos2])
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(obj_positions[:, 0], obj_positions[:, 1])
    x, y = obj_positions.mean(0)
    r = np.arctan(slope)
    while r > np.pi:
      r -= np.pi

    h = pb_obj_generation.generateRoof([x, y, roof_z],
                                       pb.getQuaternionFromEuler(
                                         [0., 0., r]),
                                       npr.uniform(self.block_scale_range[0], self.block_scale_range[1]))
    self.objects.append(h)
    self.object_types[h] = constants.ROOF
    self.structure_objs.append(h)
    self.wait(50)




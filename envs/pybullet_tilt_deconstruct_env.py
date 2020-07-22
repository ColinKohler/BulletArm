import pybullet as pb
import numpy as np
import scipy
import numpy.random as npr
from copy import deepcopy
from helping_hands_rl_envs.simulators.pybullet.utils import pybullet_util

from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv, NoValidPositionException
from helping_hands_rl_envs.envs.pybullet_deconstruct_env import PyBulletDeconstructEnv
from helping_hands_rl_envs.envs.pybullet_tilt_env import PyBulletTiltEnv
import helping_hands_rl_envs.simulators.pybullet.utils.object_generation as pb_obj_generation
from helping_hands_rl_envs.simulators import constants

class PyBulletTiltDeconstructEnv(PyBulletDeconstructEnv, PyBulletTiltEnv):
  def __init__(self, config):
    super().__init__(config)
    self.pick_offset = 0.0
    self.place_offset = 0.015
    self.prev_obj_pos = ()

  def takeAction(self, action):
    self.prev_obj_pos = self.getObjectPositions(omit_hold=False)
    super().takeAction(action)

  def isSimValid(self):
    curr_obj_pos = self.getObjectPositions(omit_hold=False)
    dist = np.linalg.norm(curr_obj_pos - self.prev_obj_pos, axis=1)
    return (dist > 0.005).sum() == 1 and super().isSimValid()

  def _getObservation(self, action=None):
    ''''''
    old_heightmap = self.heightmap
    self.heightmap = self._getHeightmap()

    if action is None or self._isHolding() == True:
      in_hand_img = self.getEmptyInHand()
    else:
      motion_primative, x, y, z, rot = self._decodeAction(action)
      if self.action_sequence.find('z') == -1:
        # z is set for a placing action, here eliminate the place offset and put the pick offset
        z = z - self.place_offset - self.pick_offset
      in_hand_img = self.getInHandImage(self.heightmap, x, y, z, rot, old_heightmap)


    return self._isHolding(), in_hand_img, self.heightmap.reshape([self.heightmap_size, self.heightmap_size, 1])


  # def initialize(self):
  #   super().initialize()
  #   self.tilt_plain_id = -1
  #   self.tilt_plain2_id = -1

  def reset(self):
    super().reset()
    self.resetTilt()

  def generateS(self):
    padding = self.max_block_size * 1.5
    while True:
      pos = self._getValidPositions(padding, 0, [], 1)[0]
      if self.isPosOffTilt(pos):
        break
    rot = pb.getQuaternionFromEuler([0., 0., 2 * np.pi * np.random.random_sample()])
    for i in range(self.num_obj):
      handle = pb_obj_generation.generateCube((pos[0], pos[1], i * self.max_block_size + self.max_block_size / 2),
                                              rot,
                                              npr.uniform(self.block_scale_range[0], self.block_scale_range[1]))
      self.objects.append(handle)
      self.object_types[handle] = constants.CUBE
      self.structure_objs.append(handle)
    self.wait(50)

  def generateH1(self):
    padding = self.max_block_size * 1.5
    while True:
      pos = self._getValidPositions(padding, 0, [], 1)[0]
      if self.isPosOffTilt(pos):
        break
    rot = pb.getQuaternionFromEuler([0., 0., 2 * np.pi * np.random.random_sample()])
    for i in range(self.num_obj-1):
      handle = pb_obj_generation.generateCube((pos[0], pos[1], i*self.max_block_size+self.max_block_size/2),
                                              rot,
                                              npr.uniform(self.block_scale_range[0], self.block_scale_range[1]))
      self.objects.append(handle)
      self.object_types[handle] = constants.CUBE
      self.structure_objs.append(handle)
    handle = pb_obj_generation.generateTriangle(
      (pos[0], pos[1], (self.num_obj-1) * self.max_block_size + self.max_block_size / 2),
      rot,
      npr.uniform(self.block_scale_range[0], self.block_scale_range[1]))
    self.objects.append(handle)
    self.object_types[handle] = constants.TRIANGLE
    self.structure_objs.append(handle)
    self.wait(50)

  def generateH3(self):
    padding = self.max_block_size * 1.5
    while True:
      pos1 = self._getValidPositions(padding, 0, [], 1)[0]
      if self.isPosOffTilt(pos1):
        break
    min_dist = 2.1 * self.max_block_size
    max_dist = 2.2 * self.max_block_size
    sample_range = [[pos1[0] - max_dist, pos1[0] + max_dist],
                    [pos1[1] - max_dist, pos1[1] + max_dist]]
    for i in range(1000):
      try:
        pos2 = self._getValidPositions(padding, min_dist, [pos1], 1, sample_range=sample_range)[0]
      except NoValidPositionException:
        continue
      dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
      if min_dist < dist < max_dist and self.isPosOffTilt(pos2):
        break

    self.generateObject((pos1[0], pos1[1], self.max_block_size / 2),
                        pb.getQuaternionFromEuler([0., 0., 2 * np.pi * np.random.random_sample()]),
                        constants.CUBE)

    self.generateObject((pos2[0], pos2[1], self.max_block_size / 2),
                        pb.getQuaternionFromEuler([0., 0., 2 * np.pi * np.random.random_sample()]),
                        constants.CUBE)

    obj_positions = np.array([pos1, pos2])
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(obj_positions[:, 0], obj_positions[:, 1])
    x, y = obj_positions.mean(0)
    r = np.arctan(slope)
    r -= np.pi/2
    while r > np.pi:
      r -= np.pi
    while r < 0:
      r += np.pi

    self.generateObject([x, y, self.max_block_size * 1.5],
                        pb.getQuaternionFromEuler([0., 0., r]),
                        constants.BRICK)

    self.generateObject([x, y, self.max_block_size * 2.5],
                        pb.getQuaternionFromEuler([0., 0., r]),
                        constants.ROOF)
    self.wait(50)

  def generateH4(self):
    padding = self.max_block_size * 1.5
    while True:
      pos1 = self._getValidPositions(padding, 0, [], 1)[0]
      if self.isPosOffTilt(pos1):
        break
    min_dist = 2.1 * self.max_block_size
    max_dist = 2.2 * self.max_block_size
    sample_range = [[pos1[0] - max_dist, pos1[0] + max_dist],
                    [pos1[1] - max_dist, pos1[1] + max_dist]]
    for i in range(1000):
      try:
        pos2 = self._getValidPositions(padding, min_dist, [pos1], 1, sample_range=sample_range)[0]
      except NoValidPositionException:
        continue
      dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
      if min_dist < dist < max_dist and self.isPosOffTilt(pos2):
        break

    self.generateObject((pos1[0], pos1[1], self.max_block_size / 2),
                        pb.getQuaternionFromEuler([0., 0., 2 * np.pi * np.random.random_sample()]),
                        constants.CUBE)

    self.generateObject((pos2[0], pos2[1], self.max_block_size / 2),
                        pb.getQuaternionFromEuler([0., 0., 2 * np.pi * np.random.random_sample()]),
                        constants.CUBE)

    obj_positions = np.array([pos1, pos2])
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(obj_positions[:, 0], obj_positions[:, 1])
    x, y = obj_positions.mean(0)
    r = np.arctan(slope)
    r -= np.pi/2
    while r > np.pi:
      r -= np.pi
    while r < 0:
      r += np.pi

    self.generateObject([x, y, self.max_block_size * 1.5],
                        pb.getQuaternionFromEuler([0., 0., r]),
                        constants.BRICK)

    self.generateObject((pos1[0], pos1[1], self.max_block_size * 2.5),
                        pb.getQuaternionFromEuler([0., 0., 2 * np.pi * np.random.random_sample()]),
                        constants.CUBE)

    self.generateObject((pos2[0], pos2[1], self.max_block_size * 2.5),
                        pb.getQuaternionFromEuler([0., 0., 2 * np.pi * np.random.random_sample()]),
                        constants.CUBE)

    self.generateObject([x, y, self.max_block_size * 3.5],
                        pb.getQuaternionFromEuler([0., 0., r]),
                        constants.ROOF)
    self.wait(50)

  def addRandomObj(self, n):
    existing_pos = self.getObjectPositions()[:, :2].tolist()
    padding = pybullet_util.getPadding(constants.RANDOM, self.max_block_size)
    min_distance = pybullet_util.getMinDistance(constants.RANDOM, self.max_block_size)
    for j in range(100):
      other_pos = self._getValidPositions(padding, min_distance, existing_pos, n)
      if all(map(lambda p: self.isPosDistToTiltValid(p, constants.RANDOM), other_pos)):
        break
    orientations = []
    existing_pos.extend(deepcopy(other_pos))
    for position in other_pos:
      y1, y2 = self.getY1Y2fromX(position[0])
      if position[1] > y1:
        d = (position[1] - y1) * np.cos(self.tilt_rz)
        position.append(self.tilt_z1 + 0.02 + np.tan(self.tilt_plain_rx) * d)
        orientations.append(pb.getQuaternionFromEuler([self.tilt_plain_rx, 0, self.tilt_rz]))
      elif position[1] < y2:
        d = (y2 - position[1]) * np.cos(self.tilt_rz)
        position.append(self.tilt_z2 + 0.02 + np.tan(-self.tilt_plain2_rx) * d)
        orientations.append(pb.getQuaternionFromEuler([self.tilt_plain2_rx, 0, self.tilt_rz]))
      else:
        position.append(0.02)
        orientations.append(pb.getQuaternionFromEuler([0, 0, np.random.random() * np.pi * 2]))
    self._generateShapes(constants.RANDOM, n, random_orientation=False, pos=other_pos, rot=orientations, z_scale=np.random.choice([constants.z_scale_1, constants.z_scale_2]))

  def generateImproviseH3(self):
    # TODO: fix this

    lower_z1 = 0.01 * 0.5
    lower_z2 = 0.01 * 1.5
    hier_z = 0.02 * 0.5
    roof_z = 0.02 + 0.015

    lower_z1 = self.max_block_size * 0.25
    lower_z2 = self.max_block_size * 0.75
    hier_z = self.max_block_size * 0.5
    roof_z = self.max_block_size * 1.5

    def generate(pos, zscale=1, rz=None):
      if rz is None:
        rz = 2 * np.pi * np.random.random_sample()
      if zscale == 1:
        zscale = constants.z_scale_1
      elif zscale == 2:
        zscale = constants.z_scale_2
      handle = pb_obj_generation.generateRandomObj(pos,
                                                   pb.getQuaternionFromEuler(
                                                     [0., 0., rz]),
                                                   npr.uniform(self.block_scale_range[0], self.block_scale_range[1]),
                                                   zscale)
      self.objects.append(handle)
      self.object_types[handle] = constants.RANDOM
      self.structure_objs.append(handle)

    padding = self.max_block_size * 1.5
    while True:
      pos1 = self._getValidPositions(padding, 0, [], 1)[0]
      if self.isPosOffTilt(pos1, min_dist=0.03):
        break
    min_dist = 2.1 * self.max_block_size
    max_dist = 2.4 * self.max_block_size
    sample_range = [[pos1[0] - max_dist, pos1[0] + max_dist],
                    [pos1[1] - max_dist, pos1[1] + max_dist]]
    for i in range(1000):
      try:
        pos2 = self._getValidPositions(padding, min_dist, [pos1], 1, sample_range=sample_range)[0]
      except NoValidPositionException:
        continue
      dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
      if min_dist < dist < max_dist and self.isPosOffTilt(pos2, min_dist=0.03):
        break

    obj_positions = np.array([pos1, pos2])
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(obj_positions[:, 0], obj_positions[:, 1])
    x, y = obj_positions.mean(0)
    r = np.arctan(slope)
    r -= np.pi/2
    while r > np.pi:
      r -= np.pi
    while r < 0:
      r += np.pi

    t = np.random.choice(4)
    if t == 0:
      generate([pos1[0], pos1[1], lower_z1], 1, r)
      generate([pos1[0], pos1[1], lower_z2], 1, r)
      generate([pos2[0], pos2[1], lower_z1], 1, r)
      generate([pos2[0], pos2[1], lower_z2], 1, r)

    elif t == 1:
      generate([pos1[0], pos1[1], lower_z1], 1, r)
      generate([pos1[0], pos1[1], lower_z2], 1, r)
      generate([pos2[0], pos2[1], hier_z], 2, r)
      self.addRandomObj(1)

    elif t == 2:
      generate([pos1[0], pos1[1], hier_z], 2, r)
      generate([pos2[0], pos2[1], lower_z1], 1, r)
      generate([pos2[0], pos2[1], lower_z2], 1, r)
      self.addRandomObj(1)

    elif t == 3:
      generate([pos1[0], pos1[1], hier_z], 2, r)
      generate([pos2[0], pos2[1], hier_z], 2, r)
      self.addRandomObj(2)

    h = pb_obj_generation.generateRoof([x, y, roof_z],
                                       pb.getQuaternionFromEuler(
                                         [0., 0., r]),
                                       npr.uniform(self.block_scale_range[0], self.block_scale_range[1]))
    self.objects.append(h)
    self.object_types[h] = constants.ROOF
    self.structure_objs.append(h)
    self.wait(50)

  def generateImproviseH4(self):
    # TODO: fix this
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
    pos1 = self._getValidPositions(padding, 0, [], 1, sample_range=[self.workspace[0], [self.tilt_border2+0.02, self.tilt_border-0.02]])[0]
    min_dist = 1.7 * self.max_block_size
    max_dist = 2.4 * self.max_block_size
    sample_range = [[pos1[0] - max_dist, pos1[0] + max_dist],
                    [max(pos1[1] - max_dist, self.tilt_border2+0.02), min(pos1[1] + max_dist, self.tilt_border-0.02)]]
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

  def generateImproviseH5(self):
    # TODO: fix this
    roof_z = 4.4*0.014 + 0.5*0.03
    def generate(pos, scale=0.6, zscale=1, rz=None):
      if rz is None:
        rz = 2 * np.pi * np.random.random_sample()
      handle = pb_obj_generation.generateRandomObj(pos,
                                                   pb.getQuaternionFromEuler(
                                                     [0., 0., rz]),
                                                   scale,
                                                   zscale)
      self.objects.append(handle)
      self.object_types[handle] = constants.RANDOM
      self.structure_objs.append(handle)

    padding = self.max_block_size * 1.5
    while True:
      pos1 = self._getValidPositions(padding, 0, [], 1)[0]
      if self.isPosOffTilt(pos1, min_dist=0.03):
        break
    min_dist = 2.2 * self.max_block_size
    max_dist = 2.4 * self.max_block_size
    sample_range = [[pos1[0] - max_dist, pos1[0] + max_dist],
                    [pos1[1] - max_dist, pos1[1] + max_dist]]
    for i in range(1000):
      try:
        pos2 = self._getValidPositions(padding, min_dist, [pos1], 1, sample_range=sample_range)[0]
      except NoValidPositionException:
        continue
      dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
      if min_dist < dist < max_dist and self.isPosOffTilt(pos2, min_dist=0.03):
        break

    base1_zscale1 = np.random.uniform(2, 2.2)
    base1_zscale2 = np.random.uniform(2, 2.2)

    base2_zscale1 = np.random.uniform(2, 2.2)
    base2_zscale2 = np.random.uniform(2, 2.2)

    base1_scale1 = np.random.uniform(0.5, 0.7)
    base1_scale2 = np.random.uniform(0.5, base1_scale1)

    base2_scale1 = np.random.uniform(0.5, 0.7)
    base2_scale2 = np.random.uniform(0.5, base2_scale1)


    base1_zscale1 = 0.6 * base1_zscale1 / base1_scale1
    base1_zscale2 = 0.6 * base1_zscale2 / base1_scale2

    base2_zscale1 = 0.6 * base2_zscale1 / base2_scale1
    base2_zscale2 = 0.6 * base2_zscale2 / base2_scale2



    obj_positions = np.array([pos1, pos2])
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(obj_positions[:, 0], obj_positions[:, 1])
    x, y = obj_positions.mean(0)
    r = np.arctan(slope)
    r -= np.pi / 2
    while r > np.pi:
      r -= np.pi
    while r < 0:
      r += np.pi

    generate([pos1[0], pos1[1], base1_zscale1 * 0.007], base1_scale1, base1_zscale1, r)
    generate([pos1[0], pos1[1], base1_zscale1 * 0.014 + base1_zscale2 * 0.007], base1_scale2, base1_zscale2, r)

    generate([pos2[0], pos2[1], base2_zscale1 * 0.007], base2_scale1, base2_zscale1, r)
    generate([pos2[0], pos2[1], base2_zscale1 * 0.014 + base2_zscale2 * 0.007], base2_scale2, base2_zscale2, r)



    h = pb_obj_generation.generateRoof([x, y, roof_z],
                                       pb.getQuaternionFromEuler(
                                         [0., 0., r]),
                                       npr.uniform(self.block_scale_range[0], self.block_scale_range[1]))
    self.objects.append(h)
    self.object_types[h] = constants.ROOF
    self.structure_objs.append(h)
    self.wait(100)

  def generateImproviseH6(self):
    def generateRandom(pos, scale=0.6, zscale=1, rz=None):
      if rz is None:
        rz = 2 * np.pi * np.random.random_sample()
      handle = pb_obj_generation.generateRandomObj(pos,
                                                   pb.getQuaternionFromEuler(
                                                     [0., 0., rz]),
                                                   scale,
                                                   zscale)
      self.objects.append(handle)
      self.object_types[handle] = constants.RANDOM
      self.structure_objs.append(handle)

    def generateBrick(pos, rz=None, x_scale=0.6, y_scale=0.6, z_scale=0.6):
      if rz is None:
        rz = 2 * np.pi * np.random.random_sample()
      handle = pb_obj_generation.generateRandomBrick(pos,
                                                     pb.getQuaternionFromEuler(
                                                       [0., 0., rz]),
                                                     x_scale, y_scale, z_scale)
      self.objects.append(handle)
      self.object_types[handle] = constants.BRICK
      self.structure_objs.append(handle)

    padding = self.max_block_size * 1.5
    while True:
      pos1 = self._getValidPositions(padding, 0, [], 1)[0]
      if self.isPosOffTilt(pos1, min_dist=0.03):
        break
    min_dist = 2.7 * self.max_block_size
    max_dist = 2.8 * self.max_block_size
    sample_range = [[pos1[0] - max_dist, pos1[0] + max_dist],
                    [pos1[1] - max_dist, pos1[1] + max_dist]]
    for i in range(1000):
      try:
        pos2 = self._getValidPositions(padding, min_dist, [pos1], 1, sample_range=sample_range)[0]
      except NoValidPositionException:
        continue
      dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
      if min_dist < dist < max_dist and self.isPosOffTilt(pos2, min_dist=0.03):
        break

    base1_zscale = np.random.uniform(2, 2.2)
    base2_zscale = np.random.uniform(2, 2.2)
    base1_scale = np.random.uniform(0.6, 0.9)
    base2_scale = np.random.uniform(0.6, 0.9)
    base1_zscale = 0.6 * base1_zscale / base1_scale
    base2_zscale = 0.6 * base2_zscale / base2_scale

    brick_xscale = np.random.uniform(0.5, 0.7)
    brick_yscale = np.random.uniform(0.5, 0.7)
    brick_zscale = np.random.uniform(0.4, 0.7)


    obj_positions = np.array([pos1, pos2])
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(obj_positions[:, 0], obj_positions[:, 1])
    x, y = obj_positions.mean(0)
    r = np.arctan(slope)
    r -= np.pi / 2
    while r > np.pi:
      r -= np.pi
    while r < 0:
      r += np.pi


    generateRandom([pos1[0], pos1[1], base1_zscale * 0.007], base1_scale, base1_zscale, r)
    generateRandom([pos2[0], pos2[1], base1_zscale * 0.007], base2_scale, base2_zscale, r)

    generateBrick([x, y, self.max_block_size * 1.5], r, brick_xscale, brick_yscale, brick_zscale)

    self.generateObject([x, y, self.max_block_size * 2.5],
                        pb.getQuaternionFromEuler([0., 0., r]),
                        constants.ROOF)
    self.wait(100)
    pass

  def generateImproviseH2(self):
    def generateRandom(pos, scale=0.6, zscale=1, rz=None):
      if rz is None:
        rz = 2 * np.pi * np.random.random_sample()
      handle = pb_obj_generation.generateRandomObj(pos,
                                                   pb.getQuaternionFromEuler(
                                                     [0., 0., rz]),
                                                   scale,
                                                   zscale)
      self.objects.append(handle)
      self.object_types[handle] = constants.RANDOM
      self.structure_objs.append(handle)

    padding = self.max_block_size * 1.5
    while True:
      pos1 = self._getValidPositions(padding, 0, [], 1)[0]
      if self.isPosOffTilt(pos1, min_dist=0.03):
        break
    min_dist = 2.7 * self.max_block_size
    max_dist = 2.8 * self.max_block_size
    sample_range = [[pos1[0] - max_dist, pos1[0] + max_dist],
                    [pos1[1] - max_dist, pos1[1] + max_dist]]
    for i in range(1000):
      try:
        pos2 = self._getValidPositions(padding, min_dist, [pos1], 1, sample_range=sample_range)[0]
      except NoValidPositionException:
        continue
      dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
      if min_dist < dist < max_dist and self.isPosOffTilt(pos2, min_dist=0.03):
        break

    base1_zscale = np.random.uniform(2, 2.2)
    base2_zscale = np.random.uniform(2, 2.2)
    base1_scale = np.random.uniform(0.6, 0.9)
    base2_scale = np.random.uniform(0.6, 0.9)
    base1_zscale = 0.6 * base1_zscale / base1_scale
    base2_zscale = 0.6 * base2_zscale / base2_scale

    obj_positions = np.array([pos1, pos2])
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(obj_positions[:, 0], obj_positions[:, 1])
    x, y = obj_positions.mean(0)
    r = np.arctan(slope)
    r -= np.pi / 2
    while r > np.pi:
      r -= np.pi
    while r < 0:
      r += np.pi


    generateRandom([pos1[0], pos1[1], base1_zscale * 0.007], base1_scale, base1_zscale, r)
    generateRandom([pos2[0], pos2[1], base1_zscale * 0.007], base2_scale, base2_zscale, r)

    self.generateObject([x, y, self.max_block_size * 1.5],
                        pb.getQuaternionFromEuler([0., 0., r]),
                        constants.ROOF)
    self.wait(100)
    pass

  def generateObject(self, pos, rot, obj_type):
    if obj_type == constants.CUBE:
      handle = pb_obj_generation.generateCube(pos,
                                              rot,
                                              npr.uniform(self.block_scale_range[0], self.block_scale_range[1]))
    elif obj_type == constants.BRICK:
      handle = pb_obj_generation.generateBrick(pos,
                                               rot,
                                               npr.uniform(self.block_scale_range[0], self.block_scale_range[1]))
    elif obj_type == constants.ROOF:
      handle = pb_obj_generation.generateRoof(pos,
                                              rot,
                                              npr.uniform(self.block_scale_range[0], self.block_scale_range[1]))
    elif obj_type == constants.TRIANGLE:
      handle = pb_obj_generation.generateTriangle(pos,
                                                  rot,
                                                  npr.uniform(self.block_scale_range[0], self.block_scale_range[1]))
    elif obj_type == constants.RANDOM:
      handle = pb_obj_generation.generateRandomObj(pos,
                                                   rot,
                                                   npr.uniform(self.block_scale_range[0], self.block_scale_range[1]))
    self.objects.append(handle)
    self.object_types[handle] = obj_type
    self.structure_objs.append(handle)




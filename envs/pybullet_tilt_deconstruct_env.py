import pybullet as pb
import numpy as np
import scipy
import numpy.random as npr

from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv, NoValidPositionException
from helping_hands_rl_envs.envs.pybullet_deconstruct_env import PyBulletDeconstructEnv
from helping_hands_rl_envs.envs.pybullet_tilt_env import PyBulletTiltEnv
import helping_hands_rl_envs.simulators.pybullet.utils.object_generation as pb_obj_generation
from helping_hands_rl_envs.simulators import constants

class PyBulletTiltDeconstructEnv(PyBulletDeconstructEnv, PyBulletTiltEnv):
  def __init__(self, config):
    super().__init__(config)
    self.pick_offset = 0.01
    self.place_offset = 0.02

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
    while True:
      pos1 = self._getValidPositions(padding, 0, [], 1)[0]
      if self.isPosOffTilt(pos1):
        break
    min_dist = 1.7 * self.max_block_size
    max_dist = 2.4 * self.max_block_size
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
    r -= np.pi/2
    while r > np.pi:
      r -= np.pi
    while r < 0:
      r += np.pi

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




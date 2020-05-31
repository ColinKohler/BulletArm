import pybullet as pb
import numpy as np
import scipy
import numpy.random as npr

from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv, NoValidPositionException
from helping_hands_rl_envs.envs.pybullet_deconstruct_env import PyBulletDeconstructEnv
from helping_hands_rl_envs.envs.pybullet_tilt_env import PyBulletTiltEnv
import helping_hands_rl_envs.simulators.pybullet.utils.object_generation as pb_obj_generation
from helping_hands_rl_envs.simulators import constants

class PyBulletTiltDeconstructEnv(PyBulletDeconstructEnv):
  def __init__(self, config):
    super().__init__(config)

    self.rx_range = (0, np.pi / 6)
    self.tilt_plain_rx = 0
    self.tilt_plain2_rx = 0
    self.tilt_plain_id = -1
    self.tilt_plain2_id = -1
    self.pick_rx = 0
    self.tilt_border = 0.035
    self.tilt_border2 = -0.035

  def _getObservation(self, action=None):
    ''''''
    old_heightmap = self.heightmap

    image_arr = pb.getCameraImage(width=self.heightmap_size, height=self.heightmap_size,
                                  viewMatrix=self.view_matrix, projectionMatrix=self.proj_matrix)
    self.heightmap = image_arr[3] - np.min(image_arr[3])

    if action is None or self._isHolding() == True:
      in_hand_img = self.getEmptyInHand()
    else:
      motion_primative, x, y, z, rot = self._decodeAction(action)
      # z is set for a placing action, here eliminate the place offset and put the pick offset
      z = z - self.place_offset - self.pick_offset
      in_hand_img = self.getInHandImage(self.heightmap, x, y, z, rot, old_heightmap)


    return self._isHolding(), in_hand_img, self.heightmap.reshape([self.heightmap_size, self.heightmap_size, 1])


  def resetTilt(self):
    if self.tilt_plain_id > -1:
      pb.removeBody(self.tilt_plain_id)
    if self.tilt_plain2_id > -1:
      pb.removeBody(self.tilt_plain2_id)

    self.tilt_plain_rx = (self.rx_range[1] - self.rx_range[0]) * np.random.random_sample() + self.rx_range[0]
    self.tilt_plain_id = pb.loadURDF('plane.urdf',
                                     [0.5 * (self.workspace[0][1] + self.workspace[0][0]), self.tilt_border, 0],
                                     pb.getQuaternionFromEuler([self.tilt_plain_rx, 0, 0]),
                                     globalScaling=0.005)
    self.tilt_plain2_rx = (self.rx_range[0] - self.rx_range[1]) * np.random.random_sample() + self.rx_range[0]
    self.tilt_plain2_id = pb.loadURDF('plane.urdf',
                                      [0.5 * (self.workspace[0][1] + self.workspace[0][0]), self.tilt_border2, 0],
                                      pb.getQuaternionFromEuler([self.tilt_plain2_rx, 0, 0]),
                                      globalScaling=0.005)

  def generateH1(self):
    self.resetTilt()
    padding = self.max_block_size * 1.5
    pos = self._getValidPositions(padding, 0, [], 1, sample_range=[self.workspace[0], [self.tilt_border2+0.02, self.tilt_border-0.02]])[0]
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




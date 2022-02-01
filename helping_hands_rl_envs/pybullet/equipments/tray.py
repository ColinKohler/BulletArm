import os
import pybullet as pb
import numpy as np

import helping_hands_rl_envs
from helping_hands_rl_envs.pybullet.utils import pybullet_util

class Tray:
  def __init__(self):
    self.root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
    self.id = None

  def initialize(self, pos=(0,0,0), rot=(0,0,0,1), size=(0.2, 0.2, 0.2), color=[0.8, 0.8, 0.8, 1]):
    '''
    :param pos:
    :param rot:
    :param size:
    :param color:
    :return: tray with walls have inclination, where inclination is the angle between walls and ground.
             Box: inclination 90
             Plate: inclination 0
    '''
    inclination = np.pi * (45/180)
    cos_offset = np.cos(inclination)
    sin_offset = np.sin(inclination)
    half_wall_height = size[2] / (2 * sin_offset)
    half_thickness = 0.005
    size0 = size[0] / 2 + 2 * half_wall_height * cos_offset
    size1 = size[1] / 2 + 2 * half_wall_height * cos_offset
    bottom_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[size0, size1, half_thickness], rgbaColor=color)
    bottom_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[size0, size1, half_thickness])

    front_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[half_thickness, size1, half_wall_height], rgbaColor=color)
    front_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[half_thickness, size1, half_wall_height])

    back_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[half_thickness, size1, half_wall_height], rgbaColor=color)
    back_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[half_thickness, size1, half_wall_height])

    left_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[size0, half_thickness, half_wall_height], rgbaColor=color)
    left_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[size0, half_thickness, half_wall_height])

    right_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[size0, half_thickness, half_wall_height], rgbaColor=color)
    right_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[size0, half_thickness, half_wall_height])


    self.id = pb.createMultiBody(baseMass=0,
                                 baseCollisionShapeIndex=bottom_collision,
                                 baseVisualShapeIndex=bottom_visual,
                                 basePosition=pos,
                                 baseOrientation=rot,
                                 linkMasses=[1, 1, 1, 1],
                                 linkCollisionShapeIndices=[front_collision, back_collision, left_collision, right_collision],
                                 linkVisualShapeIndices=[front_visual, back_visual, left_visual, right_visual],
                                 linkPositions=[[-size[0]/2 - cos_offset * size[2]/2, 0, size[2]/2],
                                                [ size[0]/2 + cos_offset * size[2]/2, 0, size[2]/2],
                                                [0, -size[1]/2 - cos_offset * size[2]/2, size[2]/2],
                                                [0,  size[1]/2 + cos_offset * size[2]/2, size[2]/2]],
                                 linkOrientations=[pb.getQuaternionFromEuler([0., -inclination, 0.]),
                                                   pb.getQuaternionFromEuler([ 0., inclination, 0.]),
                                                   pb.getQuaternionFromEuler([ inclination, 0., 0.]),
                                                   pb.getQuaternionFromEuler([-inclination, 0., 0.])],
                                 linkInertialFramePositions=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                 linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
                                 linkParentIndices=[0, 0, 0, 0],
                                 linkJointTypes=[pb.JOINT_FIXED, pb.JOINT_FIXED, pb.JOINT_FIXED, pb.JOINT_FIXED],
                                 linkJointAxis=[[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
    )

    pb.changeDynamics(self.id,
                      -1,
                      rollingFriction=0.01,
                      linearDamping=0.1)

  def reset(self, pos=(0,0,0), rot=(0,0,0,1)):
    pb.resetBasePositionAndOrientation(self.id, pos, rot)

  def remove(self):
    if self.id:
      pb.removeBody(self.id)
    self.id = None

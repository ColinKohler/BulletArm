import os
import pybullet as pb
import numpy as np

import bulletarm
from bulletarm.pybullet.utils.pybullet_util import constants
from bulletarm.pybullet.utils import transformations

BOX = 0
CYLINDER = 1

class DrawerHandle:
  def __init__(self, drawer_id, fixed=True):
    self.drawer_id = drawer_id
    self.drawer_fw_id = 6
    if fixed:
      self.grip_type = 1
      self.grip_to_drawer = 0.05
      self.grip_length = 0.1
      self.grip_width = 0.015
      self.sidebar_type = 1
      self.sidebar_dist = 0.07
      self.sidebar_width = 0.01
    else:
      self.grip_type = np.random.choice(2)
      grip_to_drawer_range = [0.03, 0.05]
      grip_length_range = [0.1, 0.15]
      grip_width_range = [0.01, 0.02]
      self.grip_to_drawer = np.random.random() * (grip_to_drawer_range[1] - grip_to_drawer_range[0]) + grip_to_drawer_range[0]
      self.grip_length = np.random.random() * (grip_length_range[1] - grip_length_range[0]) + grip_length_range[0]
      self.grip_width = np.random.random() * (grip_width_range[1] - grip_width_range[0]) + grip_width_range[0]
      # self.grip_to_drawer = 0.03
      # self.grip_length = 0.1
      # self.grip_width = 0.02

      sidebar_dist_range = [0.07, self.grip_length]
      sidebar_width_range = [0.01, 0.02]

      self.sidebar_type = np.random.choice(2)
      self.sidebar_dist = np.random.random() * (sidebar_dist_range[1] - sidebar_dist_range[0]) + sidebar_dist_range[0]
      self.sidebar_width = np.random.random() * (sidebar_width_range[1] - sidebar_width_range[0]) + sidebar_width_range[0]
      # self.sidebar_dist = 0.07
      # self.sidebar_width = 0.02

    self.sidebar_length = self.grip_to_drawer

    grip_position = list(pb.getLinkState(drawer_id, self.drawer_fw_id)[4])
    grip_position[0] -= self.grip_to_drawer

    if self.grip_type == BOX:
      grip_visual_id = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[self.grip_width/2, self.grip_length/2, self.grip_width/2], rgbaColor=[1, 1, 1, 1])
      grip_collision_id = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[self.grip_width/2, self.grip_length/2, self.grip_width/2])
    elif self.grip_type == CYLINDER:
      grip_visual_id = pb.createVisualShape(pb.GEOM_CYLINDER, length=self.grip_length, radius=self.grip_width/2, rgbaColor=[1, 1, 1, 1], visualFrameOrientation=pb.getQuaternionFromEuler([np.pi/2, 0, 0]))
      grip_collision_id = pb.createCollisionShape(pb.GEOM_CYLINDER, height=self.grip_length, radius=self.grip_width/2, collisionFrameOrientation=pb.getQuaternionFromEuler([np.pi/2, 0, 0]))
    else:
      raise NotImplementedError

    if self.sidebar_type == BOX:
      left_visual_id = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[self.sidebar_length / 2, self.sidebar_width / 2,
                                                                      self.sidebar_width / 2], rgbaColor=[1, 1, 1, 1])

      right_visual_id = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[self.sidebar_length / 2, self.sidebar_width / 2,
                                                                       self.sidebar_width / 2], rgbaColor=[1, 1, 1, 1])
    elif self.sidebar_type == CYLINDER:
      left_visual_id = pb.createVisualShape(pb.GEOM_CYLINDER, length=self.sidebar_length, radius=self.sidebar_width/2, rgbaColor=[1, 1, 1, 1], visualFrameOrientation=pb.getQuaternionFromEuler([0, np.pi/2, 0]))

      right_visual_id = pb.createVisualShape(pb.GEOM_CYLINDER, length=self.sidebar_length, radius=self.sidebar_width/2, rgbaColor=[1, 1, 1, 1], visualFrameOrientation=pb.getQuaternionFromEuler([0, np.pi/2, 0]))
    else:
      raise NotImplementedError

    self.id = pb.createMultiBody(0.1, grip_collision_id, grip_visual_id,
                                 basePosition = grip_position,
                                 baseOrientation = pb.getQuaternionFromEuler([0, 0, 0]),
                                 linkMasses=[1, 1],
                                 linkCollisionShapeIndices=[left_visual_id, right_visual_id],
                                 linkVisualShapeIndices=[left_visual_id, right_visual_id],
                                 linkPositions=[[self.sidebar_length / 2, self.sidebar_dist / 2, 0],
                                                [self.sidebar_length / 2, -self.sidebar_dist / 2, 0]],
                                 linkOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
                                 linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
                                 linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
                                 linkParentIndices=[0, 0],
                                 linkJointTypes=[pb.JOINT_FIXED, pb.JOINT_FIXED],
                                 linkJointAxis=[[0, 0, 1], [0, 0, 1]])

    pb.changeDynamics(self.id, -1,
                      lateralFriction=200,
                      spinningFriction=0,
                      rollingFriction=0)

    pb.createConstraint(drawer_id, self.drawer_fw_id, self.id, -1,
                        jointType=pb.JOINT_FIXED, jointAxis=[0, 0, 0],
                        parentFramePosition=[self.grip_to_drawer, 0, 0],
                        childFramePosition=[0, 0, 0],
                        childFrameOrientation=pb.getQuaternionFromEuler([0, 0, np.pi]))

    for _ in range(100):
      pb.stepSimulation()
    pass

  def reset(self):
    drawer_fw_pos = pb.getLinkState(self.drawer_id, self.drawer_fw_id)[0]
    drawer_fw_rot = pb.getLinkState(self.drawer_id, self.drawer_fw_id)[1]
    m = np.array(pb.getMatrixFromQuaternion(drawer_fw_rot)).reshape(3, 3)
    pos = np.array(drawer_fw_pos) + m[:, 0] * self.grip_to_drawer
    # rot_matrix = m @ np.array(transformations.euler_matrix(0, 0, np.pi))[:3, :3]
    rot_matrix = np.eye(4)
    rot_matrix[:3, :3] = m @ np.array(transformations.euler_matrix(0, 0, np.pi))[:3, :3]
    pb.resetBasePositionAndOrientation(self.id, pos, transformations.quaternion_from_matrix(rot_matrix))

  def getPosition(self):
    pos, rot = pb.getBasePositionAndOrientation(self.id)
    m = np.array(pb.getMatrixFromQuaternion(rot)).reshape(3, 3)
    offset = 0.01
    pos = pos + m[:, 0] * offset
    # pos = list(pb.getBasePositionAndOrientation(self.id)[0])
    #
    # pos[0] += 0.01
    return pos

  def getRotation(self):
    pos, rot = pb.getBasePositionAndOrientation(self.id)
    return rot

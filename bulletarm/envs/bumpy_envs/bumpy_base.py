import pybullet as pb
from bulletarm.envs.base_env import BaseEnv
import numpy as np
import os
import bulletarm
from bulletarm.pybullet.utils import constants


class BumpyBase:
  def __init__(self):
    self.bump_rs = [0, 0, 0, 0]
    self.bump_ids = []
    self.bump_ext = self.workspace_size/2

    self.platform_id = -1

    # self.bump_max_angle = np.pi/8
    self.bump_max_angle = np.deg2rad(15)
    # self.bump_max_angle = 0
    self.bump_offset = self.bump_ext/2 * np.tan(self.bump_max_angle)
    self.object_init_z += self.bump_offset

  def initialize(self):
    self.bump_ids = []
    self.platform_id = -1

  def resetBumps(self):
    for i in self.bump_ids:
      pb.removeBody(i)
    self.bump_ids = []

    self.bump_rs = [np.random.random() * self.bump_max_angle for _ in range(9)]
    # self.bump_rs = [np.pi/6 for _ in range(4)]

    bump_xs = np.linspace(-self.workspace_size/2, self.workspace_size/2, 5)[np.array([0, 2, 4])]
    bump_ys = np.linspace(-self.workspace_size/2, self.workspace_size/2, 5)[np.array([0, 2, 4])]
    bump_poses = np.array(np.meshgrid(bump_xs, bump_ys)).T.reshape(-1, 2)
    bump_rz = np.random.random() * np.pi
    R = np.array([[np.cos(bump_rz), -np.sin(bump_rz)],
                  [np.sin(bump_rz), np.cos(bump_rz)]])

    bump_poses = R.dot(bump_poses.T).T
    bump_poses += self.workspace[:2].mean(1)

    obj_pattern = os.path.join(constants.URDF_PATH, 'pyramid/pyramid.obj')
    for i in range(9):
      bump_visual_shape = pb.createVisualShape(shapeType=pb.GEOM_MESH, fileName=obj_pattern, meshScale=[self.bump_ext, self.bump_ext, np.tan(self.bump_rs[i])*self.bump_ext*0.5])
      bump_collision_shape = pb.createCollisionShape(shapeType=pb.GEOM_MESH, fileName=obj_pattern, meshScale=[self.bump_ext, self.bump_ext, np.tan(self.bump_rs[i])*self.bump_ext*0.5])
      bump_id = pb.createMultiBody(baseMass=0,
                                   baseInertialFramePosition=[0, 0, 0],
                                   baseCollisionShapeIndex=bump_collision_shape,
                                   baseVisualShapeIndex=bump_visual_shape,
                                   basePosition=[bump_poses[i, 0], bump_poses[i, 1], 0],
                                   baseOrientation=pb.getQuaternionFromEuler((0, 0, bump_rz)))
      pb.changeDynamics(bump_id, -1, lateralFriction=100, linearDamping=0.04, angularDamping=0.04, restitution=0,
                        contactStiffness=3000, contactDamping=100)
      self.bump_ids.append(bump_id)

  def resetPlatform(self, pos, rz, size):
    if self.platform_id != -1:
      pb.removeBody(self.platform_id)
    pf_visual_shape = pb.createVisualShape(shapeType=pb.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, self.bump_offset/2])
    pf_collision_shape = pb.createCollisionShape(shapeType=pb.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, self.bump_offset/2])
    self.platform_id = pb.createMultiBody(baseMass=0,
                                 baseInertialFramePosition=[0, 0, 0],
                                 baseCollisionShapeIndex=pf_collision_shape,
                                 baseVisualShapeIndex=pf_visual_shape,
                                 basePosition=[pos[0], pos[1], self.bump_offset/2],
                                 baseOrientation=pb.getQuaternionFromEuler((0, 0, rz)))

  def isObjOnPlatform(self, obj):
    return obj.isTouchingId(self.platform_id)

import os
import pybullet as pb
import numpy as np

from bulletarm.pybullet.utils import transformations
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.equipments.drawer_handle import DrawerHandle
from bulletarm.pybullet.objects.pybullet_object import PybulletObject
from typing import List

class Drawer:
  def __init__(self, model_id=1):
    assert model_id in [1, 2]
    self.model_id = model_id
    self.id = None
    self.handle = None
    self.object_init_link_id = 5
    self.cids = []

  def initialize(self, pos=(0,0,0), rot=(0,0,0,1), scale=0.5):
    drawer_urdf_filepath = os.path.join(constants.URDF_PATH, 'drawer{}.urdf'.format(self.model_id))
    self.id = pb.loadURDF(drawer_urdf_filepath, pos, rot, globalScaling=scale)
    self.handle = DrawerHandle(self.id)

  def remove(self):
    if self.id:
      pb.removeBody(self.id)
    if self.handle:
      pb.removeBody(self.handle.id)
    self.id = None
    self.handle = None

  def isObjInsideDrawer(self, obj):
    contact_points = obj.getContactPoints()
    for p in contact_points:
      if p[2] == self.id and p[4] == 4:
        return True
    return False

  def getPose(self):
    pos, rot = pb.getBasePositionAndOrientation(self.id)
    return list(pos), list(rot)

  def getHandlePosition(self):
    # return pb.getLinkState(self.handle.id, 0)[4]
    return self.handle.getPosition()

  def getHandleRotation(self):
    return self.handle.getRotation()

  def reset(self, pos=(0,0,0), rot=(0,0,0,1)):
    pb.resetBasePositionAndOrientation(self.id, pos, rot)
    pb.resetJointState(self.id, 1, 0)
    self.handle.reset()
    for i in range(50):
      pb.stepSimulation()
    pb.resetJointState(self.id, 1, 0)
    for i in range(50):
      pb.stepSimulation()
    pass

  def isDrawerOpen(self):
    return pb.getJointState(self.id, 1)[0] > 0.15

  def isDrawerClosed(self):
    return pb.getJointState(self.id, 1)[0] < 0.02

  def getObjInitPos(self):
    return pb.getLinkState(self.id, self.object_init_link_id)[0]

  def getObjInitRot(self):
    return pb.getLinkState(self.id, self.object_init_link_id)[1]

  def constraintObjects(self, objects: List[PybulletObject]):
    drawer_pos, drawer_rot = self.getObjInitPos(), self.getObjInitRot()
    wTd = transformations.quaternion_matrix(drawer_rot)
    wTd[:3, 3] = drawer_pos
    self.cids = []
    for obj in objects:
      pos, rot = obj.getPose()
      wTo = transformations.quaternion_matrix(rot)
      wTo[:3, 3] = pos
      dTo = np.linalg.inv(wTd).dot(wTo)
      cid = pb.createConstraint(self.id, self.object_init_link_id, obj.object_id, -1,
                                jointType=pb.JOINT_FIXED, jointAxis=[0, 0, 0],
                                parentFramePosition=dTo[:3, 3],
                                childFramePosition=[0, 0, 0],
                                parentFrameOrientation=transformations.quaternion_from_matrix(dTo))
      self.cids.append(cid)

  def releaseObjectConstraints(self):
    for cid in self.cids:
      pb.removeConstraint(cid)

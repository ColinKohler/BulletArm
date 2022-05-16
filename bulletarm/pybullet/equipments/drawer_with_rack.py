import pybullet as pb
import numpy as np

from bulletarm.pybullet.equipments.drawer import Drawer
from bulletarm.pybullet.equipments.rack import Rack

class DrawerWithRack:
  def __init__(self, rack_n=3):
    self.drawer = Drawer(model_id=2)
    self.rack = Rack(rack_n, dist=0.05)

  def initialize(self, pos=(0,0,0), rot=(0,0,0,1)):
    self.drawer.initialize(pos, rot)
    self.rack.initialize(self.drawer.getObjInitPos(), rot)
    pb.createConstraint(self.drawer.id, self.drawer.object_init_link_id, self.rack.ids[0], -1,
                        jointType = pb.JOINT_FIXED, jointAxis = [0, 0, 0],
                        parentFramePosition = [0, 0, 0],
                        childFramePosition = [0, 0, 0],
                        childFrameOrientation = pb.getQuaternionFromEuler([0, 0, np.pi]))
  def reset(self, pos=(0,0,0), rot=(0,0,0,1)):
    pb.resetBasePositionAndOrientation(self.drawer.id, pos, rot)
    pb.resetJointState(self.drawer.id, 1, 0)
    self.drawer.handle.reset()
    self.rack.reset(self.drawer.getObjInitPos(), rot)
    for i in range(50):
      pb.stepSimulation()


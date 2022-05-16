import copy
import numpy as np
import pybullet as pb

from bulletarm.pybullet.robots.kuka import Kuka

class KukaFloatPick(Kuka):
  def __init__(self):
    super().__init__()

  def pick(self, pos, rot, offset, dynamic=True, objects=None, simulate_grasp=True):
    ''''''
    # Setup pre-grasp pos and default orientation
    self.openGripper()
    pre_pos = copy.copy(pos)
    m = np.array(pb.getMatrixFromQuaternion(rot)).reshape(3, 3)
    pre_pos -= m[:, 2] * offset
    # rot = pb.getQuaternionFromEuler([np.pi/2.,-np.pi,np.pi/2])
    pre_rot = rot

    # Move to pre-grasp pose and then grasp pose
    self.moveTo(pre_pos, pre_rot, dynamic)
    if simulate_grasp:
      self.moveTo(pos, rot, True)
      # Grasp object and lift up to pre pose
      gripper_fully_closed = self.closeGripper()
      if gripper_fully_closed:
        self.openGripper()
        self.moveTo(pre_pos, pre_rot, dynamic)
      else:
        self.moveTo(pre_pos, pre_rot, True)
        self.adjustGripperCommand()
        for i in range(10):
          pb.stepSimulation()
        self.holding_obj = self.getPickedObj(objects)
      self.moveToJ(self.home_positions_joint, dynamic)
      self.checkGripperClosed()

    else:
      self.moveTo(pos, rot, dynamic)
      self.holding_obj = self.getPickedObj(objects)

      endToObj = self.getEndToHoldingObj()
      endToObj = np.abs(endToObj)
      if endToObj is None or (endToObj.max(1) < 0.95).any():
        self.holding_obj = None
      self.moveToJ(self.home_positions_joint, dynamic)

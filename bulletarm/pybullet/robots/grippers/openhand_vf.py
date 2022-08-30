import pybullet as pb

class OpenHandVF(object):
  ''' OpenHand's VF model gripper. '''
  def __init__(self):
    pass

  def initialize(self, robot_id):
    pass

  def getFingerForce(self):
    pass

  def controlGripper(self, open_ratio, max_it=100):
    pass

  def openGripper(self):
    pass

  def closeGripper(self):
    pass

  def getGripperOpenRatio(self):
    pass

  def getGripperImg(self, img_size, workspace_size, obs_size_m):
    pass

  def adjustGripperCommand(self):
    pass

  def checkGripperClosed(self):
    pass

  def gripperHasForce(self):
    pass

  def _getGripperJointPosition(self):
    pass

  def _sendGripperCommand(self, target_pos1, target_pos2):
    pass

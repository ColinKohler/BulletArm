import pybullet as pb

class OpenHandVF(object):
  ''' OpenHand's VF model gripper. '''
  def __init__(self):
    self.end_effector_index = 9
    self.home_positions = [0., 0.]

  def initialize(self, robot_id):
    self.id = robot_id

  def enableFingerForceTorqueSensors(self):
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

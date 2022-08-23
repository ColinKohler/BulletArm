from bulletarm.pybullet.robots.ur5 import Ur5
from bulletarm.pybullet.robots.kuka import Kuka
from bulletarm.pybullet.robots.panda import Panda

from bulletarm.pybullet.robots.ur5 import COMPATABLE_GRIPPERS as ur5_grippers

def createRobot(robot, gripper):
  if robot == 'kuka':
    return Kuka()
  elif robot == 'panda':
    return Panda()
  elif robot == 'ur5':
    gripper = gripper if gripper else 'robotiq'
    if gripper not in ur5_grippers:
      raise ValueError('Invalid gripper passed for UR5. Valid grippers: {}'.format(ur5_grippers))
    return UR5(gripper)
  else:
    raise ValueError('Invalid robot passed to robot factory.')

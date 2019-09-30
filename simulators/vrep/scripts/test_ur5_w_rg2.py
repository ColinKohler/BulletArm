import sys
import time
import numpy as np

sys.path.append('..')
from simulation import vrep
from robots.ur5 import UR5
from grippers.rg2 import RG2

VREP_BLOCKING = vrep.simx_opmode_blocking

def main():
  # Attempt to connect to simulator
  vrep.simxFinish(-1)
  sim_client = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
  if sim_client == -1:
    print 'Failed to connect to simulation. Exiting...'
    exit()
  else:
    print 'Connected to simulation.'

  # Create UR5 and restart simulator
  gripper = RG2(sim_client)
  ur5 = UR5(sim_client, gripper)
  vrep.simxStopSimulation(sim_client, VREP_BLOCKING)
  time.sleep(2)
  vrep.simxStartSimulation(sim_client, VREP_BLOCKING)
  time.sleep(2)

  # Move arm to random positon and close/open gripper
  target_pose = np.eye(4)
  target_pose[:3,-1] =  [-0.0, 0.2, 0.2]
  ur5.moveTo(target_pose)
  ur5.closeGripper()
  ur5.openGripper()

  # Wait for arm to move the exit
  time.sleep(1)
  vrep.simxStopSimulation(sim_client, VREP_BLOCKING)
  exit()

if __name__ == '__main__':
  main()

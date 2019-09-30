import sys
import time
import numpy as np
import numpy.random as npr

sys.path.append('./')
sys.path.append('..')
from simulation import vrep
from robots.ur5 import UR5
from grippers.rg2 import RG2
import utils.vrep_utils as utils
from utils import transformations

VREP_BLOCKING = vrep.simx_opmode_blocking

def main():
  # Attempt to connect to simulator
  sim_client = utils.connectToSimulation('127.0.0.1', 19997)

  # Create UR5 and restart simulator
  gripper = RG2(sim_client)
  ur5 = UR5(sim_client, gripper)
  vrep.simxStopSimulation(sim_client, VREP_BLOCKING)
  time.sleep(2)
  vrep.simxStartSimulation(sim_client, VREP_BLOCKING)
  time.sleep(2)

  # Generate a cube
  position = [0., 0.5, 0.15]
  orientation = [0, 0, 0]
  size = [0.05, 0.05, 0.05]
  mass = 0.1
  color = [255, 0, 0]
  cube = utils.generateShape(sim_client, 'cube', 0, size, position, orientation, mass, color)
  time.sleep(2)

  # Execute pick on cube
  pose = transformations.euler_matrix(np.radians(90), 0, np.radians(90))
  pose[:3,-1] = [0, 0.5, 0.0]
  offset = 0.2
  ur5.pick(pose, offset)

  pose = transformations.euler_matrix(np.radians(90), 0, np.radians(90))
  pose = transformations.euler_matrix(np.pi/2, 0.0, np.pi/2)
  pose[:3,-1] = [0.0, 0.5, 0.5]
  ur5.moveTo(pose)

  pose = transformations.euler_matrix(np.radians(90), 0, np.radians(90))
  pose = transformations.euler_matrix(np.pi/2, 0.0, np.pi/2)
  pose[:3,-1] = [0.25, 0.25, 0.0]
  ur5.place(pose, offset)

  # Wait for arm to move the exit
  time.sleep(1)
  vrep.simxStopSimulation(sim_client, VREP_BLOCKING)
  exit()

if __name__ == '__main__':
  main()

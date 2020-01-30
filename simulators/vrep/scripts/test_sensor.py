import sys
import time
import numpy as np
import pptk
import matplotlib.pyplot as plt

sys.path.append('./')
sys.path.append('..')
from simulation import vrep
from robots.ur5 import UR5
from grippers.rg2 import RG2
from sensors.vision_sensor import VisionSensor
import utils.vrep_utils as utils

VREP_BLOCKING = vrep.simx_opmode_blocking

def main():
  # Attempt to connect to simulator
  sim_client = utils.connectToSimulation('127.0.0.1', 19997)

  # Create UR5 and sensor and restart simulator
  gripper = RG2(sim_client)
  ur5 = UR5(sim_client, gripper)
  cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
  workspace = np.asarray([[-0.25, 0.25], [0.25, 0.75], [0, 0.2]])
  sensor = VisionSensor(sim_client, 'Vision_sensor_persp', workspace, cam_intrinsics)
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

  # Get sensor data and display it
  depth_pc, rgb_pc = sensor.getPointCloud()
  v = pptk.viewer(depth_pc)
  v.attributes(rgb_pc)

  depth_heightmap, rgb_heightmap = sensor.getHeightmap()
  plt.imshow(depth_heightmap)
  plt.show()

  plt.imshow(rgb_heightmap)
  plt.show()

  # Exit
  vrep.simxStopSimulation(sim_client, VREP_BLOCKING)
  exit()

if __name__ == '__main__':
  main()

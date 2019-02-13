import os
import time
import numpy as np

from vrep_arm_toolkit.utils import transformations
from vrep_arm_toolkit.simulation import vrep

VREP_BLOCKING = vrep.simx_opmode_blocking
VREP_ONESHOT = vrep.simx_opmode_oneshot_wait
VREP_CHILD_SCRIPT = vrep.sim_scripttype_childscript

OBJ_FLOAT_PARAM_BBOX_MAX_Z = 26

#------------------------------------------------------------------------------------------------#
#                                       Simulation Control                                       #
#------------------------------------------------------------------------------------------------#
# Connects to the simulation at the given address and port
def connectToSimulation(ip_address, port):
  sim_client = vrep.simxStart(ip_address, port, True, True, 10000, 5)
  if sim_client == -1:
    print('Failed to connect to simulation. Exiting...')
    exit()
  else:
    print('Connected to simluation')

  return sim_client

# Disconnect to the simulation
def disconnectToSimulation(sim_client):
  vrep.simxFinish(sim_client)

# Stop the V-Rep simulator
def stopSimulation(sim_client):
  vrep.simxStopSimulation(sim_client, VREP_BLOCKING)

# Restart the V-Rep simulator and get the various object handles
def restartSimulation(sim_client):
  vrep.simxStopSimulation(sim_client, VREP_BLOCKING)
  time.sleep(1)
  vrep.simxStartSimulation(sim_client, VREP_BLOCKING)
  time.sleep(1)

#------------------------------------------------------------------------------------------------#
#                                         Scrip API calls                                        #
#------------------------------------------------------------------------------------------------#

def generateShape(sim_client, name, shape_type, size, position,
                  orientation, mass, color=[255., 255., 255.]):
  '''
  Attempts to generate the desired shape in the vrep simulation. This requires the 'createShape' function
  to be in the 'remoteAPICommandServer' dummy object in the vrep simulation. See 'simulation/sensor_example.ttt'.

  Args:
    - sim_client: vrep client object attached to the desired simulation
    - name: Name of the object to be created
    - shape_type: Type of shape to be generate (0: cuboid, 1: sphere, 2: cylinder, 3: cone)
    - size: List of 3 floats describing the shape size
    - position: List of [x, y, z] floats to generate object at
    - orientation: List of euler floats describing object orientation
    - mass: The mass of the object to generate
    - color: Color of the shape to generate

  Returns: object handle if successful, None otherwise
  '''
  sim_ret = vrep.simxCallScriptFunction(sim_client, 'remoteApiCommandServer', VREP_CHILD_SCRIPT,
                                        'createShape', [shape_type],
                                        size + position + orientation + color + [mass], [name],
                                        bytearray(), VREP_BLOCKING)
  if sim_ret[0] == 8:
    return None
  else:
    return sim_ret[1][0]

def importShape(sim_client, name, mesh_file, position, orientation, color=[255., 255., 255.]):
  '''
  Attempts to import the desired mesh in the vrep simulation. This requires the 'importShape' function
  to be in the 'remoteAPICommandServer' dummy object in the vrep simulation. See 'simulation/sensor_example.ttt'.

  Args:
    - sim_client: vrep client object attached to the desired simulation
    - name: Name of the object to be created
    - mesh_file: Path to the mesh to be imported
    - position: List of [x, y, z] floats to generate object at
    - orientation: List of euler floats describing object orientation
    - color: Color of the shape to generate

  Returns: object handle if successful, None otherwise
  '''
  sim_ret = vrep.simxCallScriptFunction(sim_client, 'remoteApiCommandServer', VREP_CHILD_SCRIPT,
                                        'importShape', [], position + orientation + color,
                                        [mesh_file, name], bytearray(), VREP_BLOCKING)
  if sim_ret[0] == 8:
    return None
  else:
    return sim_ret[1][0]

#------------------------------------------------------------------------------------------------#
#                                       Object Helpers                                           #
#------------------------------------------------------------------------------------------------#

# Returns the objects handle
def getObjectHandle(sim_client, obj_name):
  return vrep.simxGetObjectHandle(sim_client, obj_name, VREP_BLOCKING)

# Returns object pose as 4x4 transformation matrix
def getObjectPose(sim_client, obj_handle):
  sim_ret, obj_position = getObjectPosition(sim_client, obj_handle)
  sim_ret, obj_orientation = getObjectOrientation(sim_client, obj_handle)

  obj_pose = transformations.euler_matrix(obj_orientation[0], obj_orientation[1], obj_orientation[2])
  obj_pose[:3,-1] = np.asarray(obj_position)

  return sim_ret, obj_pose

# Sets object to a given pose
def setObjectPose(sim_client, obj_handle, pose):
  pass

# Returns object position as numpy array
def getObjectPosition(sim_client, obj_handle):
  sim_ret, obj_position = vrep.simxGetObjectPosition(sim_client, obj_handle, -1, VREP_BLOCKING)
  return sim_ret, np.asarray(obj_position)

# Sets an object to a given position
def setObjectPosition(sim_client, obj_handle, position):
  return vrep.simxSetObjectPosition(sim_client, obj_handle, -1, position, VREP_BLOCKING)

# Returns the object orientation as numpy array
def getObjectOrientation(sim_client, obj_handle):
  sim_ret, orientation = vrep.simxGetObjectOrientation(sim_client, obj_handle, -1, VREP_BLOCKING)
  return sim_ret, np.asarray(orientation)

# Sets an object to a given orientation
def setObjectOrientation(sim_client, obj_handle, orientation):
  return vrep.simxSetObjectOrientation(sim_client, obj_handle, -1, orientation, VREP_BLOCKING)

def getObjectMaxZ(sim_client, obj_handle):
  return vrep.simxGetObjectFloatParameter(sim_client, obj_handle, OBJ_FLOAT_PARAM_BBOX_MAX_Z, VREP_BLOCKING)

#------------------------------------------------------------------------------------------------#
#                                       Joint Helpers                                            #
#------------------------------------------------------------------------------------------------#

# Gets the joint position
def getJointPosition(sim_client, joint):
  sim_ret, position = vrep.simxGetJointPosition(sim_client, joint, VREP_BLOCKING)
  return sim_ret, position

# Sets a joints force to a given force
def setJointForce(sim_client, joint, force):
  return vrep.simxSetJointForce(sim_client, joint, force, VREP_BLOCKING)

# Sets a joints target velocity to a given velocity
def setJointTargetVelocity(sim_client, joint, velocity):
  return vrep.simxSetJointTargetVelocity(sim_client, joint, velocity, VREP_BLOCKING)

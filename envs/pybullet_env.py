import time
import numpy as np
import numpy.random as npr

import pybullet as pb
import pybullet_data

# Motion primatives
PICK_PRIMATIVE = 0
PLACE_PRIMATIVE = 1
PUSH_PRIMATIVE = 2

class PyBulletEnv(object):
  ''''''
  def __init__(self):
    # Connect to pybullet and add data files to path
    self.client = pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Environment specific variables
    self._timestep = 1. / 240.
    self.home_positions = [0., 0., -2.137, 1.432, -0.915, -1.591, 0.071, 0., 0., 0.,
                           0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    self.workspace = np.asarray([[0.5, 0.75],
                                 [-0.25, 0.25],
                                 [0, 0.4]])

    self.img_size = 250
    self.view_matrix = pb.computeViewMatrixFromYawPitchRoll([0.5, 0.0, 0], 1.0, 180, 0, 0, 1)
    self.proj_matrix = pb.computeProjectionMatrix(-0.25, 0.25, -0.25, 0.25, -1.0, 10.0)

  def reset(self):
    ''''''
    pb.resetSimulation()
    pb.setTimeStep(self._timestep)

    pb.setGravity(0, 0, -10)
    self.table_id = pb.loadURDF('plane.urdf', [0,0,0])

    # Load the UR5 and set it to the home positions
    self.ur5_id = pb.loadURDF('urdf/ur5/ur5_w_robotiq_85_gripper.urdf', [0,0,0], [0,0,0,1])
    self.num_joints = pb.getNumJoints(self.ur5_id)
    [pb.resetJointState(self.ur5_id, idx, self.home_positions[idx]) for idx in range(self.num_joints)]

    # Load test block
    block_id = pb.loadURDF('block.urdf', [0.5, 0, 0])

    # Step simulation
    pb.stepSimulation()

    for idx in range(self.num_joints):
      print('{}: {}'.format(idx, pb.getJointInfo(self.ur5_id, idx)[1]))

    # target_pos = [0.5, 0.0, 0.2]
    # target_rot = pb.getQuaternionFromEuler([np.pi/2.,-np.pi,np.pi/2])
    # ik_solve = pb.calculateInverseKinematics(self.ur5_id, 6, target_pos, target_rot)
    # for i in range(1,7):
    #   pb.setJointMotorControl2(bodyIndex=self.ur5_id, jointIndex=i, controlMode=pb.POSITION_CONTROL, targetPosition=ik_solve[i-1], targetVelocity=0, force=500, positionGain=0.03, velocityGain=1)
    # pb.resetJointState(self.ur5_id, 1, ik_solve[0])
    # pb.resetJointState(self.ur5_id, 2, ik_solve[1])
    # pb.resetJointState(self.ur5_id, 3, ik_solve[2])
    # pb.resetJointState(self.ur5_id, 4, ik_solve[3])
    # pb.resetJointState(self.ur5_id, 5, ik_solve[4])
    # pb.resetJointState(self.ur5_id, 6, ik_solve[5])

    import ipdb; ipdb.set_trace()

    return self._getState()

  def step(self, action):
    ''''''
    # motion_primative, x, y, rot = action

    # Get transfor for action
    # T = transformations.euler_matrix(np.radians(90), rot, np.radians(90))
    # T[:2,3] = [x, y]
    # T[2,3] = self._getPrimativeHeight(motion_primative, x, y)

    # Take action specfied by motion primative
    # if motion_primative == PICK_PRIMATIVE:
    #   pass
    # elif motion_primative == PLACE_PRIMATIVE:
    #   pass
    # elif motion_primative == PUSH_PRIMATIVE:
    #   pass
    # else:
    #   raise ValueError('Bad motion primative supplied for action.')

    target_pos = [0.5, 0.0, 0.25]
    target_rot = pb.getQuaternionFromEuler([np.pi/2.,-np.pi,np.pi/2])
    ik_solve = pb.calculateInverseKinematics(self.ur5_id, 6, target_pos, target_rot)
    for i in range(1,7):
      pb.setJointMotorControl2(bodyIndex=self.ur5_id, jointIndex=i, controlMode=pb.POSITION_CONTROL, targetPosition=ik_solve[i-1], targetVelocity=0, force=500, positionGain=0.03, velocityGain=1)

    # Step simulation
    pb.stepSimulation()

    return self._getState()

  def _getState(self):
    ''''''
    image_arr = pb.getCameraImage(width=self.img_size, height=self.img_size,
                                  viewMatrix=self.view_matrix, projectionMatrix=self.proj_matrix)
    depth_img = image_arr[3]

    return False, depth_img

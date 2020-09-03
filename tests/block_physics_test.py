import numpy as np
import numpy.random as npr
import pybullet as pb
import pybullet_data
import time

import helping_hands_rl_envs.simulators.pybullet.utils.object_generation as pb_obj_generation
from helping_hands_rl_envs.simulators.pybullet.robots.kuka import Kuka

npr.seed(10)
client = pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())

timestep = 1. / 240.
block_scale = 0.7
orientation = pb.getQuaternionFromEuler([0., 0., 0.])

rot = 0
rot_q = pb.getQuaternionFromEuler([0, np.pi, -rot])
while rot < -np.pi:
  rot += np.pi
while rot > 0:
  rot -= np.pi
rot_q = pb.getQuaternionFromEuler([0, np.pi, rot])

pos_1 = [0.4740994829371439, 0.09970086068207779, 0.02]
pos_2 = [0.33597223738828336, -0.035347514737952265, 0.02]
action_1= [0.4740994915994709, 0.09970135426659088, 0.026997323036193847]
action_2= [0.34597224607142874, -0.025347019866595863, 0.09199684619903564]

robot = Kuka()
pick_pre_offset = 0.10
place_pre_offset = 0.10

offset = 0.015

for _ in range(100):
  pb.resetSimulation()
  pb.setPhysicsEngineParameter(numSubSteps=0, numSolverIterations=200, solverResidualThreshold=1e-10, constraintSolverType=pb.CONSTRAINT_SOLVER_LCP_SI)
  pb.setTimeStep(timestep)
  pb.setGravity(0, 0, -10)

  table_id = pb.loadURDF('plane.urdf', [0,0,0])
  pb.changeDynamics(table_id, -1, linearDamping=0.0, angularDamping=0.0, restitution=0, contactStiffness=3000, contactDamping=100)

  robot.reset()

  pb.stepSimulation()

  cube_1 = pb_obj_generation.generateCube(pos_1, orientation, block_scale)
  [pb.stepSimulation() for _ in range(100)]
  cube_2 = pb_obj_generation.generateCube(pos_2, orientation, block_scale)
  [pb.stepSimulation() for _ in range(100)]

  robot.pick(action_1, rot_q, pick_pre_offset, dynamic=True, simulate_grasp=True)
  [pb.stepSimulation() for _ in range(100)]
  robot.place(action_2, rot_q, place_pre_offset, dynamic=True, simulate_grasp=True)

  time.sleep(1)

import pybullet as pb
import numpy as np
import numpy.random as npr

from bulletarm.pybullet.utils import constants
from bulletarm.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from bulletarm.pybullet.utils import transformations
import bulletarm.pybullet.utils.object_generation as pb_obj_generation
from bulletarm.pybullet.equipments.square_peg_hole import SquarePegHole
from bulletarm.planners.close_loop_peg_insertion_planner import CloseLoopPegInsertionPlanner

class CloseLoopPegInsertionEnv(CloseLoopEnv):
  def __init__(self, config):
    super().__init__(config)
    self.peg_hole = SquarePegHole()
    self.peg_hole_rz = 0
    self.peg_hole_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 0]

    # Modify physics to restrict object penetration during contact
    self.num_solver_iterations = 200
    self.solver_residual_threshold = 1e-7
    pb.setPhysicsEngineParameter(
      numSubSteps=0,
      numSolverIterations=self.num_solver_iterations,
      solverResidualThreshold=self.solver_residual_threshold,
      constraintSolverType=pb.CONSTRAINT_SOLVER_LCP_SI,
      contactERP=0.2,
    )

  def resetPegHole(self):
    self.peg_hole_rz = np.random.random_sample() * 2*np.pi - np.pi if self.random_orientation else 0
    self.peg_hole_pos = self._getValidPositions(0.1, 0, [], 1)[0]
    self.peg_hole_pos.append(0)
    self.peg_hole.reset(self.peg_hole_pos, pb.getQuaternionFromEuler((0, 0, self.peg_hole_rz)))

  def initialize(self):
    super().initialize()
    self.peg_hole.initialize(pos=self.peg_hole_pos, rot=pb.getQuaternionFromEuler((0, 0, self.peg_hole_rz)))

  def reset(self):
    self.resetPybulletWorkspace()
    self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.3], transformations.quaternion_from_euler(0, 0, 0))

    self.resetPegHole()
    self.peg = self._generateShapes(
      constants.SQUARE_PEG,
      pos=[[self.workspace[0].mean(), self.workspace[1].mean(), 0.27]],
      rot=[[0,0,0,1]],
      scale=0.1075,
      wait=False
    )[0]
    #pb.changeDynamics(
    #  self.peg.object_id,
    #  -1,
    #  contactStiffness=100,
    #  contactDamping=1,
    #)

    ee_pose = self.robot._getEndEffectorPose()
    peg_pose = self.peg.getPose()
    world_to_ee = pb.invertTransform(ee_pose[0], ee_pose[1])
    peg_to_ee = pb.multiplyTransforms(world_to_ee[0], world_to_ee[1], peg_pose[0], peg_pose[1])
    cid = pb.createConstraint(
      parentBodyUniqueId=self.robot.id,
      parentLinkIndex=self.robot.end_effector_index,
      childBodyUniqueId=self.peg.object_id,
      childLinkIndex=-1,
      jointType=pb.JOINT_FIXED,
      jointAxis=(0,0,0),
      parentFramePosition=peg_to_ee[0],
      parentFrameOrientation=peg_to_ee[1],
      childFramePosition=(0,0,0.0),
      childFrameOrientation=(0,0,0),
    )
    #pb.changeConstraint(cid, maxForce=50)

    self.robot.closeGripper()
    self.setRobotHoldingObj()

    return self._getObservation()

  def step(self, action):
    # Force the gripper to stay closed
    action[0] = 0.
    return super().step(action)

  def _checkTermination(self):
    if not self._isPegInHand():
      return True

    hole_pos, hole_rot = self.peg_hole.getHolePose()
    peg_pos = self.peg.getPosition()

    return np.allclose(hole_pos[:2], peg_pos[:2], atol=1e-2) and peg_pos[2] < 0.11

  def _getReward(self):
    hole_pos, hole_rot = self.peg_hole.getHolePose()
    peg_pos = self.peg.getPosition()

    success_reward = 1 if np.allclose(hole_pos[:2], peg_pos[:2], atol=1e-2) and peg_pos[2] < 0.11 else 0
    return success_reward

  def _isPegInHand(self):
    peg_pos = self.peg.getPosition()
    peg_rot = transformations.euler_from_quaternion(self.peg.getRotation())

    end_effector_pos = self.robot._getEndEffectorPosition()
    end_effector_pos[2] -= 0.03

    return np.allclose(peg_pos, end_effector_pos, atol=5e-1) and np.allclose(peg_rot[:2], [0., 0.], atol=1e-1)

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
    self.peg_scale_range = config['object_scale_range']

    self.peg_hole = SquarePegHole()
    self.peg_hole_rz = 0
    self.peg_hole_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 0]

    # Modify physics to restrict object penetration during contact
    pb.setPhysicsEngineParameter(
      numSubSteps=0,
      numSolverIterations=self.num_solver_iterations,
      solverResidualThreshold=self.solver_residual_threshold,
      constraintSolverType=pb.CONSTRAINT_SOLVER_LCP_SI,
      #contactBreakingThreshold=0.1,
    )

  def resetPegHole(self):
    self.peg_hole_rz = np.random.random_sample() * 2*np.pi - np.pi if self.random_orientation else 0
    self.peg_hole_pos = self._getValidPositions(0.25, 0, [], 1)[0]
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
      scale=0.113,#self.peg_scale_range[0],
      wait=False
    )[0]
    pb.changeDynamics(self.peg.object_id, -1, mass=0.2, contactStiffness=1000000, contactDamping=10000)

    self.robot.closeGripper()
    self.setRobotHoldingObj()

    for _ in range(2):
      pb.stepSimulation()
    self.peg.resetPose([self.workspace[0].mean(), self.workspace[1].mean(), 0.27], [0,0,0,1])

    return self._getObservation()

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

    return np.allclose(peg_pos, end_effector_pos, atol=5e-1) and np.allclose(peg_rot[:2], [0., 0.], atol=0.1)

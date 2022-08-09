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
      scale=0.11,#self.peg_scale_range[0],
      wait=False
    )[0]
    pb.changeDynamics(self.peg.object_id, -1, lateralFriction=0.75, mass=0.2)

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

    finger_a_force, finger_b_force = self.robot.getFingerForce()
    force_weight = 1e-2
    force_mag = np.sqrt(np.sum(np.array(finger_a_force)**2)) + np.sqrt(np.sum(np.array(finger_b_force)**2))
    force_pen = -(np.clip(force_mag - 2, 0, 10) / 10.0)

    success_reward = 1 if np.allclose(hole_pos[:2], peg_pos[:2], atol=1e-2) and peg_pos[2] < 0.11 else 0
    #drop_pen = 0 if self._isPegInHand() else -1
    return success_reward# + force_weight * force_pen

  def _isPegInHand(self):
    peg_pos = self.peg.getPosition()
    end_effector_pos = self.robot._getEndEffectorPosition()
    end_effector_pos[2] -= 0.03

    return np.allclose(peg_pos, end_effector_pos, atol=5e-1)

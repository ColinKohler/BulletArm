import pybullet as pb
import numpy as np
import numpy.random as npr
from collections import deque

from bulletarm.pybullet.utils import constants
from bulletarm.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from bulletarm.pybullet.utils import transformations
import bulletarm.pybullet.utils.object_generation as pb_obj_generation
from bulletarm.pybullet.equipment.square_peg_hole import SquarePegHole
from bulletarm.pybullet.equipment.round_peg_hole import RoundPegHole
from bulletarm.planners.close_loop_peg_insertion_planner import CloseLoopPegInsertionPlanner

class CloseLoopPegInsertionEnv(CloseLoopEnv):
  def __init__(self, config):
    super().__init__(config)
    if 'peg_type' not in self.config.keys():
      self.config['peg_type'] = 'square'
    self.peg_type = self.config['peg_type']

    if self.peg_type == 'round':
      self.peg_hole = RoundPegHole()
      self.peg_hole_rx = -np.pi * 0.5
      self.peg_scale = 1.45
      self.peg_type = constants.ROUND_PEG
      self.peg_insertion_height = 0.13
    elif self.peg_type == 'square':
      self.peg_hole_rx = 0
      self.peg_hole = SquarePegHole()
      self.peg_type = constants.SQUARE_PEG
      self.peg_scale = 1.18
      self.peg_insertion_height = 0.12
    self.peg_hole_rz = 0
    self.peg_hole_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 0.03]
    self.prev_ee_pos = deque(maxlen=5)
    self.prev_ee_rot = deque(maxlen=5)
    self.robot.max_torque = [50.0] * 7

  def resetPegHole(self):
    self.peg_hole_rz = np.random.random_sample() * 2*np.pi - np.pi if self.random_orientation else 0
    self.peg_hole_pos = self._getValidPositions(0.20, 0, [], 1)[0]
    self.peg_hole_pos.append(0.03)
    self.peg_hole.reset(self.peg_hole_pos, pb.getQuaternionFromEuler((self.peg_hole_rx, 0, self.peg_hole_rz)))

  def initialize(self):
    super().initialize()
    self.peg_hole.initialize(pos=self.peg_hole_pos, rot=pb.getQuaternionFromEuler((self.peg_hole_rx, 0, self.peg_hole_rz)))

  def step(self, action):
    obs, reward, done = super().step(action)
    self.prev_ee_pos.append(self.robot._getEndEffectorPosition())
    self.prev_ee_rot.append(self.robot._getEndEffectorRotation())

    return obs, reward, done

  def reset(self):
    self.resetPybulletWorkspace()
    self.robot.force_limit = None
    self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.3], transformations.quaternion_from_euler(0, 0, 0), dynamic=False)

    self.resetPegHole()
    self.peg = self._generateShapes(
      self.peg_type,
      pos=[[self.workspace[0].mean(), self.workspace[1].mean(), 0.33]],
      rot=[pb.getQuaternionFromEuler((-np.pi * 0.5, 0, 0))],
      scale=self.peg_scale, wait=False
    )[0]
    pb.changeDynamics(self.peg.object_id, -1, 1, lateralFriction=50.0, rollingFriction=0.0003, spinningFriction=0.3)
    pb.changeDynamics(self.peg.object_id, 0, 1, lateralFriction=0.1, rollingFriction=0.0003, spinningFriction=0.1)
    #pb.changeDynamics(self.peg_hole.id, 0, 1, lateralFriction=0.3, rollingFriction=0.0003, spinningFriction=0.3)

    self.robot.gripper.close()
    self.setRobotHoldingObj()

    self.stuck_steps = 10
    self.prev_ee_pos = deque(maxlen=self.stuck_steps)
    self.prev_ee_rot = deque(maxlen=self.stuck_steps)

    return self._getObservation()

  def _checkTermination(self):
    if not self._isPegInHand() or self._endEffectorStuck():
      return True

    hole_pos, hole_rot = self.peg_hole.getHolePose()
    peg_pos = self.peg.getPosition()

    return np.allclose(hole_pos[:2], peg_pos[:2], atol=1e-2) and peg_pos[2] < self.peg_insertion_height

  def _getReward(self):
    hole_pos, hole_rot = self.peg_hole.getHolePose()
    peg_pos = self.peg.getPosition()
    success_reward = 1 if np.allclose(hole_pos[:2], peg_pos[:2], atol=1e-2) and peg_pos[2] < self.peg_insertion_height else 0

    return success_reward

  def _isPegInHand(self):
    peg_pos = self.peg.getPosition()
    peg_rot = transformations.euler_from_quaternion(self.peg.getRotation())

    end_effector_pos = self.robot._getEndEffectorPosition()

    return np.allclose(peg_pos[:2], end_effector_pos[:2], atol=2e-2) and \
           np.allclose(peg_rot[:2], [-np.pi * 0.5, 0.], atol=2e-1)

  def _endEffectorStuck(self):
    return len(self.prev_ee_pos) == self.stuck_steps and \
           np.allclose(self.prev_ee_pos[0], self.prev_ee_pos, atol=1e-2) and \
           np.allclose(self.prev_ee_rot[0], self.prev_ee_rot, atol=1e-1)

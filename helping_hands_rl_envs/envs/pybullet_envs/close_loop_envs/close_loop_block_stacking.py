import pybullet as pb
import numpy as np

from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.envs.pybullet_envs.close_loop_envs.close_loop_env import CloseLoopEnv
from helping_hands_rl_envs.simulators.pybullet.utils import transformations
from helping_hands_rl_envs.planners.close_loop_block_stacking_planner import CloseLoopBlockStackingPlanner
from helping_hands_rl_envs.simulators.pybullet.utils.ortho_sensor import OrthographicSensor

class CloseLoopBlockStackingEnv(CloseLoopEnv):
  def __init__(self, config):
    super().__init__(config)
    assert self.num_obj == 2
    self.ws_size = max(self.workspace[0][1] - self.workspace[0][0], self.workspace[1][1] - self.workspace[1][0]) * 1.5
    cam_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 1]
    target_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 0]
    cam_up_vector = [-1, 0, 0]
    self.sensor = OrthographicSensor(cam_pos, cam_up_vector, target_pos, self.ws_size, 0.1, 1)

  def reset(self):
    self.resetPybulletEnv()
    self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
    self._generateShapes(constants.CUBE, self.num_obj, random_orientation=self.random_orientation)
    return self._getObservation()

  def _getValidOrientation(self, random_orientation):
    if random_orientation:
      orientation = pb.getQuaternionFromEuler([0., 0., np.pi/2 * (np.random.random_sample() - 0.5)])
    else:
      orientation = pb.getQuaternionFromEuler([0., 0., 0.])
    return orientation

  def _checkTermination(self):
    return not self._isHolding() and self._checkStack(self.objects)

  def step(self, action):
    motion_primative, x, y, z, rot = self._decodeAction(action)
    obs, reward, done = super().step(action)
    reward *= 5
    if motion_primative == constants.PICK_PRIMATIVE and not self._isHolding():
      reward -= 0.1
    reward -= np.abs(np.array([rot[-1]])).sum() * 0.1
    return obs, reward, done

  def setRobotHoldingObj(self):
    self.setRobotHoldingObjWithRotConstraint()

def createCloseLoopBlockStackingEnv(config):
  return CloseLoopBlockStackingEnv(config)

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  workspace = np.asarray([[0.2, 0.8],
                          [-0.3, 0.3],
                          [0.01, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 100, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 2, 'action_sequence': 'pxyzr', 'num_objects': 2, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'object_init_space_check': 'point', 'physics_mode': 'fast', 'object_scale_range': (1, 1), 'hard_reset_freq': 1000}
  planner_config = {'random_orientation': False}
  env_config['seed'] = 1
  env = CloseLoopBlockStackingEnv(env_config)
  planner = CloseLoopBlockStackingPlanner(env, planner_config)
  s, in_hand, obs = env.reset()

  while True:
    action = planner.getNextAction()
    obs, reward, done = env.step(action)
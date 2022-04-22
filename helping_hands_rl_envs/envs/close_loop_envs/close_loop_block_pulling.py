import pybullet as pb
import numpy as np

from helping_hands_rl_envs.pybullet.utils import constants
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from helping_hands_rl_envs.pybullet.utils import transformations
from helping_hands_rl_envs.planners.close_loop_block_pulling_planner import CloseLoopBlockPullingPlanner
from helping_hands_rl_envs.pybullet.utils.constants import NoValidPositionException

class CloseLoopBlockPullingEnv(CloseLoopEnv):
  def __init__(self, config):
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.8, 0.8]
    super().__init__(config)

  def reset(self):
    while True:
      self.resetPybulletWorkspace()
      self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
      try:
        if not self.random_orientation:
          padding = self._getDefaultBoarderPadding(constants.FLAT_BLOCK)
          min_distance = self._getDefaultMinDistance(constants.FLAT_BLOCK)
          x = np.random.random() * (self.workspace_size - padding) + self.workspace[0][0] + padding/2
          while True:
            y1 = np.random.random() * (self.workspace_size - padding) + self.workspace[1][0] + padding/2
            y2 = np.random.random() * (self.workspace_size - padding) + self.workspace[1][0] + padding/2
            if max(y1, y2) - min(y1, y2) > min_distance:
              break
          self._generateShapes(constants.FLAT_BLOCK, 2, pos=[[x, y1, self.object_init_z], [x, y2, self.object_init_z]], random_orientation=True)
        else:
          self._generateShapes(constants.FLAT_BLOCK, 2, random_orientation=self.random_orientation)
      except NoValidPositionException as e:
        continue
      else:
        break
    return self._getObservation()

  def _getValidOrientation(self, random_orientation):
    if random_orientation:
      orientation = pb.getQuaternionFromEuler([0., 0., np.pi * (np.random.random_sample() - 0.5)])
    else:
      orientation = pb.getQuaternionFromEuler([0., 0., 0.])
    return orientation

  def _checkTermination(self):
    return self.objects[0].isTouching(self.objects[1])

def createCloseLoopBlockPullingEnv(config):
  return CloseLoopBlockPullingEnv(config)

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  workspace = np.asarray([[0.2, 0.8],
                          [-0.3, 0.3],
                          [0.01, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 100, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 2, 'action_sequence': 'pxyzr', 'num_objects': 1, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'object_init_space_check': 'point', 'physics_mode': 'fast', 'object_scale_range': (1, 1), 'hard_reset_freq': 1000}
  planner_config = {'random_orientation': False}
  env_config['seed'] = 1
  env = CloseLoopBlockPullingEnv(env_config)
  planner = CloseLoopBlockPullingPlanner(env, planner_config)
  s, in_hand, obs = env.reset()
  # while True:
  #   current_pos = env.robot._getEndEffectorPosition()
  #   current_rot = transformations.euler_from_quaternion(env.robot._getEndEffectorRotation())
  #
  #   block_pos = env.objects[0].getPosition()
  #   block_rot = transformations.euler_from_quaternion(env.objects[0].getRotation())
  #
  #   pos_diff = block_pos - current_pos
  #   rot_diff = np.array(block_rot) - current_rot
  #   pos_diff[pos_diff // 0.01 > 1] = 0.01
  #   pos_diff[pos_diff // -0.01 > 1] = -0.01
  #
  #   rot_diff[rot_diff // (np.pi/32) > 1] = np.pi/32
  #   rot_diff[rot_diff // (-np.pi/32) > 1] = -np.pi/32
  #
  #   action = [1, pos_diff[0], pos_diff[1], pos_diff[2], rot_diff[2]]
  #   obs, reward, done = env.step(action)

  while True:
    action = planner.getNextAction()
    obs, reward, done = env.step(action)

  # fig, axs = plt.subplots(8, 5, figsize=(25, 40))
  # for i in range(40):
  #   action = planner.getNextAction()
  #   obs, reward, done = env.step(action)
  #   axs[i//5, i%5].imshow(obs[2][0], vmax=0.3)
  # env.reset()
  # fig.show()

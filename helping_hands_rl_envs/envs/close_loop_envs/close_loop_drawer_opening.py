import pybullet as pb
import numpy as np

from helping_hands_rl_envs.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from helping_hands_rl_envs.pybullet.utils import transformations
from helping_hands_rl_envs.planners.close_loop_drawer_opening_planner import CloseLoopDrawerOpeningPlanner
from helping_hands_rl_envs.pybullet.equipments.drawer import Drawer

class CloseLoopDrawerOpeningEnv(CloseLoopEnv):
  def __init__(self, config):
    super().__init__(config)
    self.drawer = Drawer()
    self.drawer_rot = 0

  def initialize(self):
    super().initialize()
    self.drawer.initialize((self.workspace[0].mean(), self.workspace[1].mean(), 0), pb.getQuaternionFromEuler((0, 0, 0)), 0.3)

  def reset(self):
    self.resetPybulletWorkspace()
    self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
    # pos = self._getValidPositions(0.1, 0, [], 1)[0]
    # pos.append(0)
    pos = np.array([self.workspace[0].mean(), self.workspace[1].mean(), 0])
    self.drawer_rot = np.random.random()*2*np.pi if self.random_orientation else np.random.choice([np.pi/2, 3*np.pi/2])
    m = np.array(transformations.euler_matrix(0, 0, self.drawer_rot))[:3, :3]
    dx = np.random.random() * (0.2 - 0.15) + 0.15
    dy = np.random.random() * (0.1 - -0.1) + -0.1
    pos = pos + m[:, 0] * dx
    pos = pos + m[:, 1] * dy
    self.drawer.reset(pos, transformations.quaternion_from_euler(0, 0, self.drawer_rot))

    return self._getObservation()

  def _checkTermination(self):
    return self.drawer.isDrawerOpen()

  def getObjectPoses(self, objects=None):
    obj_poses = list()

    drawer_pos, drawer_rot = self.drawer.getPose()
    drawer_rot = self.convertQuaternionToEuler(drawer_rot)
    obj_poses.append(drawer_pos + drawer_rot)
    handle_pos = list(self.drawer.getHandlePosition())
    handle_rot = list(self.drawer.getHandleRotation())
    handle_rot = self.convertQuaternionToEuler(handle_rot)
    obj_poses.append(handle_pos + handle_rot)

    return np.array(obj_poses)

def createCloseLoopDrawerOpeningEnv(config):
  return CloseLoopDrawerOpeningEnv(config)

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  workspace = np.asarray([[0.2, 0.8],
                          [-0.3, 0.3],
                          [0.01, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 100, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 2, 'action_sequence': 'pxyzr', 'num_objects': 1, 'random_orientation': True,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'object_init_space_check': 'point', 'physics_mode': 'fast', 'object_scale_range': (1.2, 1.2),
                'hard_reset_freq': 1000, }
  planner_config = {'random_orientation': True, 'dpos': 0.05, 'drot': np.pi/8}
  env_config['seed'] = 1
  env = CloseLoopDrawerOpeningEnv(env_config)
  planner = CloseLoopDrawerOpeningPlanner(env, planner_config)
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

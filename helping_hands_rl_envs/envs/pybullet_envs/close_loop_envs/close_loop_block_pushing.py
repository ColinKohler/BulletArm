import pybullet as pb
import numpy as np

from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.envs.pybullet_envs.close_loop_envs.close_loop_env import CloseLoopEnv
from helping_hands_rl_envs.simulators.pybullet.utils import transformations
from helping_hands_rl_envs.planners.close_loop_block_pushing_planner import CloseLoopBlockPushingPlanner

class CloseLoopBlockPushingEnv(CloseLoopEnv):
  def __init__(self, config):
    super().__init__(config)
    self.goal_pos = self.workspace.mean(1)[:2]
    self.goal_id = None
    self.obs_size_m = self.workspace_size * 2
    self.heightmap_resolution = self.obs_size_m / self.heightmap_size
    self.initSensor()
    self.goal_grid_size_half = 10
    self.goal_size = self.goal_grid_size_half*2 * self.heightmap_resolution

  def getGoalPixel(self):
    gripper_pos = self.robot._getEndEffectorPosition()
    goal_pixel_x = (self.goal_pos[0] - gripper_pos[0]) / self.heightmap_resolution + self.heightmap_size // 2
    goal_pixel_y = (self.goal_pos[1] - gripper_pos[1]) / self.heightmap_resolution + self.heightmap_size // 2
    return round(goal_pixel_x), round(goal_pixel_y)

  def reset(self):
    self.resetPybulletEnv()
    self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
    self._generateShapes(constants.CUBE, 1, random_orientation=self.random_orientation)
    goal_pos = self._getValidPositions(0.08, 0.09, self.getObjectPositions()[:, :2].tolist(), 1)[0]
    self.goal_pos = goal_pos

    if self.goal_id is not None:
      pb.removeBody(self.goal_id)
    goal_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[self.goal_size/2, self.goal_size/2, 0.001], rgbaColor=[0, 0, 1, 1])
    self.goal_id = pb.createMultiBody(baseMass=0,
                                      baseVisualShapeIndex=goal_visual,
                                      basePosition=[*self.goal_pos, 0],
                                      baseOrientation=transformations.quaternion_from_euler(0, 0, 0), )

    return self._getObservation()

  def _getHeightmap(self):
    heightmap = super()._getHeightmap()
    goal_x, goal_y = self.getGoalPixel()
    # heightmap[max(goal_x-self.goal_grid_size, 0):min(goal_x+self.goal_grid_size, self.heightmap_size-1), max(goal_y-self.goal_grid_size, 0):min(goal_y+self.goal_grid_size, self.heightmap_size-1)] += 0.025
    test_x = np.arange(goal_x - self.goal_grid_size_half, goal_x + self.goal_grid_size_half, 1)
    test_x = test_x[(0 <= test_x) & (test_x < 128)]
    test_y = np.arange(goal_y - self.goal_grid_size_half, goal_y + self.goal_grid_size_half, 1)
    test_y = test_y[(0 <= test_y) & (test_y < 128)]
    # heightmap[test_x, test_y] += 0.025
    X2D, Y2D = np.meshgrid(test_x, test_y)
    out = np.column_stack((X2D.ravel(), Y2D.ravel()))
    heightmap[out[:, 0].reshape(-1), out[:, 1].reshape(-1)] += 0.02
    return heightmap

  def _checkTermination(self):
    obj_pos = self.objects[0].getPosition()[:2]
    return np.linalg.norm(np.array(self.goal_pos) - np.array(obj_pos)) < 0.03

def createCloseLoopBlockPushingEnv(config):
  return CloseLoopBlockPushingEnv(config)

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
  env = CloseLoopBlockPushingEnv(env_config)
  planner = CloseLoopBlockPushingPlanner(env, planner_config)
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
    if reward > 0:
      print(1)

  # fig, axs = plt.subplots(8, 5, figsize=(25, 40))
  # for i in range(40):
  #   action = planner.getNextAction()
  #   obs, reward, done = env.step(action)
  #   axs[i//5, i%5].imshow(obs[2][0], vmax=0.3)
  # env.reset()
  # fig.show()
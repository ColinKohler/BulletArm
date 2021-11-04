import pybullet as pb
import numpy as np

from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.envs.pybullet_envs.close_loop_envs.close_loop_env import CloseLoopEnv
from helping_hands_rl_envs.simulators.pybullet.utils import transformations
from helping_hands_rl_envs.planners.close_loop_block_picking_planner import CloseLoopBlockPickingPlanner
from helping_hands_rl_envs.simulators.pybullet.equipments.tray import Tray
from helping_hands_rl_envs.simulators.constants import NoValidPositionException


class CloseLoopHouseholdPickingClutteredEnv(CloseLoopEnv):
  def __init__(self, config):
    super().__init__(config)
    self.object_init_z = 0.1
    self.obj_grasped = 0
    self.tray = Tray()
    self.bin_size = 0.4

  def initialize(self):
    super().initialize()
    self.tray.initialize(pos=[self.workspace[0].mean(), self.workspace[1].mean(), 0],
                         size=[self.bin_size + 0.03, self.bin_size + 0.03, 0.1])

  def reset(self):
    self.resetPybulletEnv()
    self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
    while True:
      try:
        for i in range(self.num_obj):
          x = (np.random.rand() - 0.5) * 0.3
          x += self.workspace[0].mean()
          y = (np.random.rand() - 0.5) * 0.3
          y += self.workspace[1].mean()
          randpos = [x, y, 0.20]
          # obj = self._generateShapes(constants.RANDOM_HOUSEHOLD, 1, random_orientation=self.random_orientation,
          #                            pos=[randpos], padding=self.min_boarder_padding,
          #                            min_distance=self.min_object_distance, model_id=-1)
          obj = self._generateShapes(constants.RANDOM_HOUSEHOLD200, 1,
                                     random_orientation=self.random_orientation,
                                     pos=[randpos], padding=0.1,
                                     min_distance=0, model_id=-1)
          pb.changeDynamics(obj[0].object_id, -1, lateralFriction=0.6)
          self.wait(10)
      except NoValidPositionException:
        continue
      else:
        break
    self.wait(200)
    self.obj_grasped = 0

    return self._getObservation()

  def step(self, action):
    pre_obj_grasped = self.obj_grasped
    obs, reward, done = super().step(action)
    if self.obj_grasped > pre_obj_grasped:
      reward = 1.0
      self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2],
                        transformations.quaternion_from_euler(0, 0, 0))
      obs = self._getObservation(action)
    return obs, reward, done

  def _getValidOrientation(self, random_orientation):
    if random_orientation:
      orientation = pb.getQuaternionFromEuler([0., 0., np.pi/2 * (np.random.random_sample() - 0.5)])
    else:
      orientation = pb.getQuaternionFromEuler([0., 0., 0.])
    return orientation

  def _checkTermination(self):
    gripper_z = self.robot._getEndEffectorPosition()[-1]
    for obj in self.objects:
      if gripper_z > 0.08 and self._isObjectHeld(obj):
        self.obj_grasped += 1
        self._removeObject(obj)
        if self.obj_grasped == self.num_obj or len(self.objects) == 0:
          return True
        return False
    return False
    # return self.robot.holding_obj == self.objects[-1] and gripper_z > 0.08

def createCloseLoopHouseholdPickingClutteredEnv(config):
  return CloseLoopHouseholdPickingClutteredEnv(config)

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  workspace = np.asarray([[0.2, 0.8],
                          [-0.3, 0.3],
                          [0.01, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 100, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 2, 'action_sequence': 'pxyzr', 'num_objects': 15, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'object_init_space_check': 'point', 'physics_mode': 'fast', 'object_scale_range': (1, 1), 'hard_reset_freq': 1000}
  planner_config = {'random_orientation': False, 'dpos': 0.05, 'drot': np.pi/8}
  env_config['seed'] = 1
  env = CloseLoopHouseholdPickingClutteredEnv(env_config)
  planner = CloseLoopBlockPickingPlanner(env, planner_config)
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
    if reward == 1:
      print(1)
    if done == 1:
      print(2)

  # fig, axs = plt.subplots(8, 5, figsize=(25, 40))
  # for i in range(40):
  #   action = planner.getNextAction()
  #   obs, reward, done = env.step(action)
  #   axs[i//5, i%5].imshow(obs[2][0], vmax=0.3)
  # env.reset()
  # fig.show()
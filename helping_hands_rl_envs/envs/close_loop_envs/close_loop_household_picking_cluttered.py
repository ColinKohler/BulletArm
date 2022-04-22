import pybullet as pb
import numpy as np

from helping_hands_rl_envs.pybullet.utils import constants
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from helping_hands_rl_envs.pybullet.utils import transformations
from helping_hands_rl_envs.planners.close_loop_household_picking_cluttered_planner import CloseLoopHouseholdPickingClutteredPlanner
from helping_hands_rl_envs.pybullet.equipments.tray import Tray
from helping_hands_rl_envs.pybullet.utils.constants import NoValidPositionException


class CloseLoopHouseholdPickingClutteredEnv(CloseLoopEnv):
  def __init__(self, config):
    if 'num_objects' not in config:
      config['num_objects'] = 15
    if 'close_loop_tray' not in config:
      config['close_loop_tray'] = True
    super().__init__(config)
    self.object_init_z = 0.1
    if 'transparent_bin' not in config:
      self.trans_bin = False
    else:
      self.trans_bin = config['transparent_bin']
    if 'collision_penalty' not in config:
      self.coll_pen = False
    else:
      self.coll_pen = config['collision_penalty']
    if 'fix_set' not in config:
      self.fix_set = False
    else:
      self.fix_set = config['fix_set']
    if 'collision_terminate' not in config:
      self.collision_terminate = False
    else:
      self.collision_terminate = config['collision_terminate']

    self.max_grasp_attempt = int(self.num_obj * 1.5)

    self.obj_grasped = 0
    self.grasp_done = 0
    self.grasp_attempted = 0
    self.current_grasp_steps = 1

  def resetEnv(self):
    self.resetPybulletWorkspace()
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
                                     min_distance=0, model_id=i+2 if self.fix_set else -1)
          pb.changeDynamics(obj[0].object_id, -1, lateralFriction=0.6)
          self.wait(10)
      except NoValidPositionException:
        continue
      else:
        break
    self.wait(200)

    self.obj_grasped = 0
    self.grasp_done = 0
    self.grasp_attempted = 0
    self.current_grasp_steps = 1

    return self._getObservation()

  def step(self, action):
    self.current_grasp_steps += 1
    pre_obj_grasped = self.obj_grasped
    obs, reward, done = super().step(action)
    if self.obj_grasped > pre_obj_grasped:
      reward = 1.0
      done = 1
    elif not self.isSimValid() or self.current_grasp_steps > self.max_steps:
      done = 1
    elif self.collision_terminate and self.robot.gripperHasForce() and not self._isHolding():
      done = 1
    else:
      done = 0
    self.grasp_done = done
    if self.coll_pen \
        and self.robot.gripperHasForce() \
        and not self._isHolding():
      reward -= 0.1
    return obs, reward, done

  def reset(self):
    self.current_grasp_steps = 1
    self.grasp_attempted += 1
    self.renderer.clearPoints()
    if not self.isSimValid() \
        or self.obj_grasped == self.num_obj \
        or len(self.objects) == 0 \
        or self.current_episode_steps == 1 \
        or self.grasp_attempted >= self.max_grasp_attempt:
      ret = self.resetEnv()
    else:
      self.robot.reset()
      self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2],
                        transformations.quaternion_from_euler(0, 0, 0))
      ret = self._getObservation()
    # TODO: set for other envs
    self.simulate_pos = self.robot._getEndEffectorPosition()
    self.simulate_rot = transformations.euler_from_quaternion(self.robot._getEndEffectorRotation())
    return ret

  def _getValidOrientation(self, random_orientation):
    if random_orientation:
      orientation = pb.getQuaternionFromEuler([0., 0., np.pi/2 * (np.random.random_sample() - 0.5)])
    else:
      orientation = pb.getQuaternionFromEuler([0., 0., 0.])
    return orientation

  def _checkTermination(self):
    gripper_z = self.robot._getEndEffectorPosition()[-1]
    for obj in self.objects:
      if gripper_z > 0.15 and self._isObjectHeld(obj):
        self.obj_grasped += 1
        self._removeObject(obj)
        if self.obj_grasped == self.num_obj or len(self.objects) == 0:
          return True
        return False
    return False
    # return self.robot.holding_obj == self.objects[-1] and gripper_z > 0.08

  def isSimValid(self):
    for obj in self.objects:
      p = obj.getPosition()
      if self._isObjectHeld(obj):
        continue
      if not self.workspace[0][0]-0.05 < p[0] < self.workspace[0][1]+0.05 and \
          self.workspace[1][0]-0.05 < p[1] < self.workspace[1][1]+0.05 and \
          self.workspace[2][0] < p[2] < self.workspace[2][1]:
        return False
    return True

def createCloseLoopHouseholdPickingClutteredEnv(config):
  return CloseLoopHouseholdPickingClutteredEnv(config)

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  workspace = np.asarray([[0.25, 0.65],
                          [-0.2, 0.2],
                          [0.01, 0.25]])
  env_config = {'workspace': workspace, 'max_steps': 100, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 2, 'action_sequence': 'pxyzr', 'num_objects': 15, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'object_init_space_check': 'point', 'physics_mode': 'fast', 'object_scale_range': (0.8, 0.8), 'hard_reset_freq': 1000}
  planner_config = {'random_orientation': False, 'dpos': 0.05, 'drot': np.pi/8}
  env_config['seed'] = 1
  env = CloseLoopHouseholdPickingClutteredEnv(env_config)
  planner = CloseLoopHouseholdPickingClutteredPlanner(env, planner_config)
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
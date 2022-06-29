from copy import deepcopy
import numpy as np
import numpy.random as npr

from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils import constants
from bulletarm.planners.waypoint_planner import WaypointPlanner

class BlockPickingEnv(BaseEnv):
  '''
  '''
  def __init__(self, config):
    # env specific parameters
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.6, 0.6]
    if 'num_objects' not in config:
      config['num_objects'] = 3
    if 'max_steps' not in config:
      config['max_steps'] = 10
    super().__init__(config)
    self.obj_grasped = 0

  def reset(self):
    ''''''
    self.resetPybulletWorkspace()
    self._generateShapes(constants.CUBE, self.num_obj, random_orientation=self.random_orientation)
    self.planner_waypoints = [
      (constants.PICK_PRIMATIVE, o, None)
      for o in self.objects
    ]

    self.obj_grasped = 0
    return self._getObservation()

  def saveState(self):
    super(BlockPickingEnv, self).saveState()
    self.state['obj_grasped'] = deepcopy(self.obj_grasped)

  def restoreState(self):
    super(BlockPickingEnv, self).restoreState()
    self.obj_grasped = self.state['obj_grasped']

  def _checkTermination(self):
    ''''''
    for obj in self.objects:
      if self._isObjectHeld(obj):
        self.obj_grasped += 1
        self._removeObject(obj)
        self.robot.holding_obj = None
        if self.obj_grasped == self.num_obj:
          return True
        return False
    return False

  def _getObservation(self, action=None):
    state, in_hand, obs = super(BlockPickingEnv, self)._getObservation()
    return 0, np.zeros_like(in_hand), obs

def createBlockPickingEnv(config):
  return BlockPickingEnv(config)

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  workspace = np.asarray([[0.25, 0.65],
                          [-0.2, 0.2],
                          [0.01, 0.25]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 128, 'render': True, 'fast_mode': False,
                'seed': 2, 'action_sequence': 'xyrp', 'num_objects': 3, 'random_orientation': True,
                'reward_type': 'sparse', 'simulate_grasp': True, 'robot': 'ur5_robotiq',
                'workspace_check': 'point', 'physics_mode': 'fast', 'object_scale_range': (0.6, 0.6), 'hard_reset_freq': 1000}
  planner_config = {'random_orientation': False, 'pos_noise': 0.0}
  env_config['seed'] = 1
  env = BlockPickingEnv(env_config)
  planner = WaypointPlanner(env, planner_config)

  for _ in range(10):
    s, in_hand, obs = env.reset()
    done = False

    while not done:
      plt.imshow(obs.squeeze(), cmap='gray'); plt.show()
      action = planner.getNextAction()
      obs, reward, done = env.step(action)
      s, in_hand, obs = obs
    plt.imshow(obs.squeeze(), cmap='gray'); plt.show()

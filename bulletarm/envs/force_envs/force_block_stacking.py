import numpy as np
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_block_stacking import CloseLoopBlockStackingEnv
from helping_hands_rl_envs.planners.close_loop_block_stacking_planner import CloseLoopBlockStackingPlanner

class ForceBlockStackingEnv(CloseLoopBlockStackingEnv):
  def __init__(self, config):
    super().__init__(config)

  def _getObservation(self, action=None):
    ''''''
    state, hand_obs, obs = super()._getObservation(action=action)

    force = np.array(self.robot.force_history)

    return state, hand_obs, obs, force

def createForceBlockStackingEnv(config):
  return ForceBlockStackingEnv(config)

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  import more_itertools
  workspace = np.asarray([[0.25, 0.65],
                          [-0.2, 0.2],
                          [0.01, 0.25]])

  env_config = {'workspace': workspace, 'max_steps': 100, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': None, 'action_sequence': 'pxyzr', 'num_objects': 3, 'random_orientation': True,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'panda',
                'object_init_space_check': 'point', 'physics_mode': 'force', 'object_scale_range': (1.0, 1.0),
                'hard_reset_freq': 1000, 'view_type': 'camera_center_xyz'}
  planner_config = {'random_orientation': True, 'dpos': 0.05, 'drot': np.pi/8}
  env = ForceBlockStackingEnv(env_config)
  planner = CloseLoopBlockStackingPlanner(env, planner_config)

  for _ in range(20):
    obs = env.reset()
    done = False
    while not done:
      action = planner.getNextAction()

      obs, reward, done = env.step(action)
      force = np.array(env.robot.force_history)

      def smooth(x, window=1):
        return np.mean(list(more_itertools.windowed(x, window)), axis=1)

      fig, ax  = plt.subplots(nrows=1, ncols=1)
      ax.plot(smooth(force[:,0]), label='F_x')
      ax.plot(smooth(force[:,1]), label='F_y')
      ax.plot(smooth(force[:,2]), label='F_z')
      ax.plot(smooth(force[:,3]), label='M_x')
      ax.plot(smooth(force[:,4]), label='M_y')
      ax.plot(smooth(force[:,5]), label='M_z')
      plt.show()

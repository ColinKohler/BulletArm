import numpy as np
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_block_pulling import CloseLoopBlockPullingEnv
from helping_hands_rl_envs.planners.close_loop_block_pulling_planner import CloseLoopBlockPullingPlanner

class ForceBlockPullingEnv(CloseLoopBlockPullingEnv):
  def __init__(self, config):
    super().__init__(config)

  def _getObservation(self, action=None):
    ''''''
    state, hand_obs, obs = super()._getObservation(action=action)
    force = np.array(self.robot.force_history)

    return state, hand_obs, obs, force

def createForceBlockPullingEnv(config):
  return ForceBlockPullingEnv(config)

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  workspace = np.asarray([[0.25, 0.65],
                          [-0.2, 0.2],
                          [0.01, 0.24]])
  env_config = {'workspace': workspace, 'max_steps': 50, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 2, 'action_sequence': 'pxyzr', 'num_objects': 1, 'random_orientation': True,
                'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'panda',
                'object_init_space_check': 'point', 'physics_mode': 'force', 'object_scale_range': (1, 1)}
  planner_config = {'random_orientation': False, 'dpos': 0.05, 'drot': np.pi/4}
  env_config['seed'] = 1
  env = ForceBlockPullingEnv(env_config)
  planner = CloseLoopBlockPullingPlanner(env, planner_config)
  s, in_hand, obs, force = env.reset()
  done = False

  while not done:
    action = planner.getNextAction()
    obs, reward, done = env.step(action)
    s, in_hand, obs, force = obs

    plt.plot(force[:,0], label='Fx')
    plt.plot(force[:,1], label='Fy')
    plt.plot(force[:,2], label='Fz')
    plt.plot(force[:,3], label='Mx')
    plt.plot(force[:,4], label='My')
    plt.plot(force[:,5], label='Mz')
    plt.legend()
    plt.show()

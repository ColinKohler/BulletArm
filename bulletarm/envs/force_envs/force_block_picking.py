import numpy as np
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_block_picking import CloseLoopBlockPickingEnv
from helping_hands_rl_envs.planners.close_loop_block_picking_planner import CloseLoopBlockPickingPlanner

class ForceBlockPickingEnv(CloseLoopBlockPickingEnv):
  def __init__(self, config):
    super().__init__(config)

  def _getObservation(self, action=None):
    ''''''
    state, hand_obs, obs = super()._getObservation(action=action)

    force = np.array(self.robot.force_history)

    return state, hand_obs, obs, force

def createForceBlockPickingEnv(config):
  return ForceBlockPickingEnv(config)

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  workspace = np.asarray([[0.25, 0.65],
                          [-0.2, 0.2],
                          [0.01, 0.25]])

  env_config = {'workspace': workspace, 'max_steps': 100, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': None, 'action_sequence': 'pxyzr', 'num_objects': 1, 'random_orientation': True,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'panda',
                'object_init_space_check': 'point', 'physics_mode': 'force', 'object_scale_range': (1.0, 1.0),
                'hard_reset_freq': 1000, 'view_type': 'camera_center_xyz'}
  planner_config = {'random_orientation': True, 'dpos': 0.05, 'drot': np.pi/8}
  env = ForceBlockPickingEnv(env_config)
  planner = CloseLoopBlockPickingPlanner(env, planner_config)

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

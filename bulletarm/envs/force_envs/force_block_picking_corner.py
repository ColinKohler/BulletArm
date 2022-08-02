import numpy as np
import more_itertools
from scipy.ndimage import uniform_filter1d
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_block_picking_corner import CloseLoopBlockPickingCornerEnv
from helping_hands_rl_envs.planners.close_loop_block_picking_corner_planner import CloseLoopBlockPickingCornerPlanner

class ForceBlockPickingCornerEnv(CloseLoopBlockPickingCornerEnv):
  def __init__(self, config):
    super().__init__(config)

  def _getObservation(self, action=None):
    ''''''
    state, hand_obs, obs = super()._getObservation(action=action)

    force = np.array(self.robot.force_history)
    #force = np.mean(list(more_itertools.windowed(force, 4, step=4)), axis=1)
    force = uniform_filter1d(force, size=16, axis=0)

    return state, hand_obs, obs, force

def createForceBlockPickingCornerEnv(config):
  return ForceBlockPickingCornerEnv(config)

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  workspace = np.asarray([[0.25, 0.65],
                          [-0.2, 0.2],
                          [0.01, 0.25]])

  env_config = {'workspace': workspace, 'max_steps': 100, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': None, 'action_sequence': 'pxyzr', 'num_objects': 1, 'random_orientation': True,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'panda',
                'object_init_space_check': 'point', 'physics_mode': 'force', 'object_scale_range': (1.2, 1.2),
                'hard_reset_freq': 1000, 'view_type': 'camera_center_xyz'}
  planner_config = {'random_orientation': True, 'dpos': 0.05, 'drot': np.pi/4}
  env = ForceBlockPickingCornerEnv(env_config)
  planner = CloseLoopBlockPickingCornerPlanner(env, planner_config)

  s, in_hand, obs, force = env.reset()
  done = False

  while not done:
    action = planner.getNextAction()
    obs, reward, done = env.step(action)
    s, in_hand, obs, force = obs

    force1 = np.clip(force, -30, 30) / 30.
    force2 = np.clip(force, -100, 100) / 100.

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].plot(force1[:,0], label='Fx')
    ax[0].plot(force1[:,1], label='Fy')
    ax[0].plot(force1[:,2], label='Fz')
    ax[0].plot(force1[:,3], label='Mx')
    ax[0].plot(force1[:,4], label='My')
    ax[0].plot(force1[:,5], label='Mz')

    ax[1].plot(force2[:,0], label='Fx')
    ax[1].plot(force2[:,1], label='Fy')
    ax[1].plot(force2[:,2], label='Fz')
    ax[1].plot(force2[:,3], label='Mx')
    ax[1].plot(force2[:,4], label='My')
    ax[1].plot(force2[:,5], label='Mz')

    plt.legend()
    plt.show()

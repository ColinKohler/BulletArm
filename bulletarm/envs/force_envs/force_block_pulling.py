import numpy as np
import numpy.random as npr
from scipy.ndimage import uniform_filter1d

from bulletarm.envs.close_loop_envs.close_loop_block_pulling import CloseLoopBlockPullingEnv

class ForceBlockPullingEnv(CloseLoopBlockPullingEnv):
  def __init__(self, config):
    super().__init__(config)

  def _getObservation(self, action=None):
    ''''''
    state, hand_obs, obs = super()._getObservation(action=action)
    force = np.array(self.robot.force_history)

    max_force = 10
    #force = np.clip(uniform_filter1d(force, size=64, axis=0), -max_force, max_force) / max_force
    force = np.clip(force, -max_force, max_force) / max_force

    return state, hand_obs, obs, force[-72:-8]

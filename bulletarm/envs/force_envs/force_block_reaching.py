import numpy as np
from bulletarm.envs.close_loop_envs.close_loop_block_reaching import CloseLoopBlockReachingEnv

class ForceBlockReachingEnv(CloseLoopBlockReachingEnv):
  def __init__(self, config):
    super().__init__(config)

  def _getObservation(self, action=None):
    ''''''
    state, hand_obs, obs = super()._getObservation(action=action)
    force = np.array(self.robot.force_history)

    return state, hand_obs, obs, force

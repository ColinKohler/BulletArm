import numpy as np
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_block_pushing import CloseLoopBlockPushingEnv

class ForceBlockPushingEnv(CloseLoopBlockPushingEnv):
  def __init__(self, config):
    super().__init__(config)

  def _getObservation(self, action=None):
    ''''''
    state, hand_obs, obs = super()._getObservation(action=action)
    force = np.array(self.robot.force_history)

    return state, hand_obs, obs, force

def createForceBlockPushingEnv(config):
  return ForceBlockPushingEnv(config)

import numpy as np
from helping_hands_rl_envs.envs.close_loop_envs.close_loop_household_picking_cluttered import CloseLoopHouseholdPickingClutteredEnv

class ForceHouseholdPickingClutteredEnv(CloseLoopHouseholdPickingClutteredEnv):
  def __init__(self, config):
    super().__init__(config)

  def _getObservation(self, action=None):
    ''''''
    state, hand_obs, obs = super()._getObservation(action=action)
    force = np.array(self.robot.force_history)

    return state, hand_obs, obs, force

def createForceHouseholdPickingClutteredEnv(config):
  return ForceHouseholdPickingClutteredEnv(config)

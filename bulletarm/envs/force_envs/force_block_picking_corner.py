import numpy as np
from bulletarm.envs.close_loop_envs.close_loop_block_picking_corner import CloseLoopBlockPickingCornerEnv

class ForceBlockPickingCornerEnv(CloseLoopBlockPickingCornerEnv):
  def __init__(self, config):
    super().__init__(config)

  def _getObservation(self, action=None):
    ''''''
    state, hand_obs, obs = super()._getObservation(action=action)
    force = np.array(self.robot.force_history)

    #force = uniform_filter1d(force, size=256, axis=0)
    #force = np.clip(force, -20, 20) / 20

    return state, hand_obs, obs, force#force[-64:]

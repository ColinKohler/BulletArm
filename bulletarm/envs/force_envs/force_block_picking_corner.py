import numpy as np
from scipy.ndimage import uniform_filter1d

from bulletarm.envs.close_loop_envs.close_loop_block_picking_corner import CloseLoopBlockPickingCornerEnv

class ForceBlockPickingCornerEnv(CloseLoopBlockPickingCornerEnv):
  def __init__(self, config):
    super().__init__(config)

  def _getObservation(self, action=None):
    ''''''
    state, hand_obs, obs = super()._getObservation(action=action)
    force = np.array(self.robot.force_history)

    max_force = 30
    force = np.clip(force, -max_force, max_force) / max_force

    return state, hand_obs, obs, force[-64:]

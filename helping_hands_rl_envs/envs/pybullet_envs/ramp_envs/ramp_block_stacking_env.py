from helping_hands_rl_envs.envs.pybullet_envs.ramp_envs.ramp_base_env import RampBaseEnv
from helping_hands_rl_envs.simulators import constants


class RampBlockStackingEnv(RampBaseEnv):
  def __init__(self, config):
    super(RampBlockStackingEnv, self).__init__(config)

  def reset(self):
    ''''''
    obj_dict = {
      constants.CUBE: self.num_obj,
    }
    self.resetWithRampAndObj(obj_dict)
    return self._getObservation()

  def _checkTermination(self):
    ''''''
    return self._checkStack()

def createRampBlockStackingEnv(config):
  return RampBlockStackingEnv(config)
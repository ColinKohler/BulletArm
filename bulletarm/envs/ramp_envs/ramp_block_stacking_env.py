from bulletarm.envs.ramp_envs.ramp_base_env import RampBaseEnv
from bulletarm.pybullet.utils import constants


class RampBlockStackingEnv(RampBaseEnv):
  def __init__(self, config):
    # env specific parameters
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.6, 0.6]
    if 'num_objects' not in config:
      config['num_objects'] = 4
    if 'max_steps' not in config:
      config['max_steps'] = 10
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

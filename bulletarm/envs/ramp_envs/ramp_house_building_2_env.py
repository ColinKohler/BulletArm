from bulletarm.envs.ramp_envs.ramp_base_env import RampBaseEnv
from bulletarm.pybullet.utils import constants


class RampHouseBuilding2Env(RampBaseEnv):
  def __init__(self, config):
    # env specific parameters
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.6, 0.6]
    if 'num_objects' not in config:
      config['num_objects'] = 3
    if 'max_steps' not in config:
      config['max_steps'] = 10
    super(RampHouseBuilding2Env, self).__init__(config)

  def reset(self):
    # super().reset()
    obj_dict = {
      constants.ROOF: 1,
      constants.CUBE: 2
    }
    self.resetWithRampAndObj(obj_dict)
    return self._getObservation()

  def _checkTermination(self):
    blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
    if self._checkOnTop(blocks[0], roofs[0]) and self._checkOnTop(blocks[1], roofs[0]):
      return True
    return False

def createRampHouseBuilding2Env(config):
  return RampHouseBuilding2Env(config)

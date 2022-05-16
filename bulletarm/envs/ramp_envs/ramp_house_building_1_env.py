from bulletarm.envs.ramp_envs.ramp_base_env import RampBaseEnv
from bulletarm.pybullet.utils import constants


class RampHouseBuilding1Env(RampBaseEnv):
  def __init__(self, config):
    # env specific parameters
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.6, 0.6]
    if 'num_objects' not in config:
      config['num_objects'] = 4
    if 'max_steps' not in config:
      config['max_steps'] = 10
    super(RampHouseBuilding1Env, self).__init__(config)

  def reset(self):
    # super().reset()
    obj_dict = {
      constants.TRIANGLE: 1,
      constants.CUBE: self.num_obj-1,
    }
    self.resetWithRampAndObj(obj_dict)
    return self._getObservation()

  def _checkTermination(self):
    ''''''
    blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
    triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))
    return self._checkStack(blocks+triangles) and self._checkObjUpright(triangles[0])

def createRampHouseBuilding1Env(config):
  return RampHouseBuilding1Env(config)

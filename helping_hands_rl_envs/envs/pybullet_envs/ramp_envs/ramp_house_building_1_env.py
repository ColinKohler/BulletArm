from helping_hands_rl_envs.envs.pybullet_envs.ramp_envs.ramp_base_env import RampBaseEnv
from helping_hands_rl_envs.simulators import constants


class RampHouseBuilding1Env(RampBaseEnv):
  def __init__(self, config):
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

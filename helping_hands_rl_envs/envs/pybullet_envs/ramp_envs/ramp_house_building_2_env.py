from helping_hands_rl_envs.envs.pybullet_envs.ramp_envs.ramp_base_env import RampBaseEnv
from helping_hands_rl_envs.simulators import constants


class RampHouseBuilding2Env(RampBaseEnv):
  def __init__(self, config):
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
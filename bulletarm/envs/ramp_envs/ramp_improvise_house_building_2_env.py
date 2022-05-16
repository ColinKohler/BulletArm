import numpy as np
from bulletarm.envs.ramp_envs.ramp_base_env import RampBaseEnv
from bulletarm.pybullet.utils import constants


class RampImproviseHouseBuilding2Env(RampBaseEnv):
  def __init__(self, config):
    # env specific parameters
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.6, 0.6]
    if 'num_objects' not in config:
      config['num_objects'] = 3
    if 'max_steps' not in config:
      config['max_steps'] = 10
    config['check_random_obj_valid'] = True
    super(RampImproviseHouseBuilding2Env, self).__init__(config)

  def generateRandomShape(self, n, poss, rots):
    for i in range(n):
      zscale = np.random.uniform(2, 2.2)
      scale = np.random.uniform(0.6, 0.9)
      zscale = 0.6 * zscale / scale
      self._generateShapes(constants.RANDOM, 1, random_orientation=False, pos=poss[i:i + 1], rot=rots[i:i + 1],
                           scale=scale, z_scale=zscale)

  def reset(self):
    obj_dict = {
      constants.ROOF: 1,
      constants.RANDOM: 2
    }
    self.resetWithRampAndObj(obj_dict)
    return self._getObservation()

  def _checkTermination(self):
    random_blocks = list(filter(lambda x: self.object_types[x] == constants.RANDOM, self.objects))
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

    if self._checkOnTop(random_blocks[0], roofs[0]) and self._checkOnTop(random_blocks[1], roofs[0]):
      return True
    return False

def createRampImproviseHouseBuilding2Env(config):
  return RampImproviseHouseBuilding2Env(config)

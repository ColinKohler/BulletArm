from bulletarm.envs.ramp_envs.ramp_base_env import RampBaseEnv
from bulletarm.pybullet.utils import constants


class RampHouseBuilding3Env(RampBaseEnv):
  def __init__(self, config):
    # env specific parameters
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.6, 0.6]
    if 'num_objects' not in config:
      config['num_objects'] = 4
    if 'max_steps' not in config:
      config['max_steps'] = 10
    super(RampHouseBuilding3Env, self).__init__(config)

  def reset(self):
    # super().reset()
    obj_dict = {
      constants.ROOF: 1,
      constants.BRICK: 1,
      constants.CUBE: 2
    }
    self.resetWithRampAndObj(obj_dict)
    return self._getObservation()

  def _checkTermination(self):
    blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
    bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

    top_blocks = []
    for block in blocks:
      if self._isObjOnTop(block, blocks):
        top_blocks.append(block)
    if len(top_blocks) != 2:
      return False
    if self._checkOnTop(top_blocks[0], bricks[0]) and \
        self._checkOnTop(top_blocks[1], bricks[0]) and \
        self._checkOnTop(bricks[0], roofs[0]) and \
        self._checkOriSimilar([bricks[0], roofs[0]]) and \
        self._checkInBetween(bricks[0], blocks[0], blocks[1]) and \
        self._checkInBetween(roofs[0], blocks[0], blocks[1]):
      return True
    return False

def createRampHouseBuilding3Env(config):
  return RampHouseBuilding3Env(config)

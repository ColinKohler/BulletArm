from helping_hands_rl_envs.envs.pybullet_envs.ramp_envs.ramp_base_env import RampBaseEnv
from helping_hands_rl_envs.simulators import constants


class RampHouseBuilding4Env(RampBaseEnv):
  def __init__(self, config):
    super(RampHouseBuilding4Env, self).__init__(config)

  def reset(self):
    # super().reset()
    obj_dict = {
      constants.ROOF: 1,
      constants.BRICK: 1,
      constants.CUBE: 4
    }
    self.resetWithRampAndObj(obj_dict)
    return self._getObservation()

  def _checkTermination(self):
    blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
    bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

    level1_blocks = list(filter(self._isObjOnGround, blocks))
    if len(level1_blocks) != 2:
      return False

    level2_blocks = list(set(blocks) - set(level1_blocks))
    return self._checkOnTop(level1_blocks[0], bricks[0]) and \
           self._checkOnTop(level1_blocks[1], bricks[0]) and \
           self._checkOnTop(bricks[0], level2_blocks[0]) and \
           self._checkOnTop(bricks[0], level2_blocks[1]) and \
           self._checkOnTop(level2_blocks[0], roofs[0]) and \
           self._checkOnTop(level2_blocks[1], roofs[0]) and \
           self._checkOriSimilar([bricks[0], roofs[0]]) and \
           self._checkInBetween(bricks[0], level1_blocks[0], level1_blocks[1]) and \
           self._checkInBetween(roofs[0], level2_blocks[0], level2_blocks[1]) and \
           self._checkInBetween(bricks[0], level2_blocks[0], level2_blocks[1])

def createRampHouseBuilding4Env(config):
  return RampHouseBuilding4Env(config)
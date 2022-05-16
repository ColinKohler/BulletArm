from bulletarm.envs.ramp_envs.ramp_deconstruct_env import RampDeconstructEnv
from bulletarm.envs.deconstruct_envs.block_stacking_deconstruct_env import BlockStackingDeconstructEnv


class RampBlockStackingDeconstructEnv(RampDeconstructEnv, BlockStackingDeconstructEnv):
  ''''''
  def __init__(self, config):
    RampDeconstructEnv.__init__(self, config)

  def generateStructure(self):
    BlockStackingDeconstructEnv.generateStructure(self)

  def checkStructure(self):
    ''''''
    return BlockStackingDeconstructEnv.checkStructure(self)


def createRampBlockStackingDeconstructEnv(config):
  return RampBlockStackingDeconstructEnv(config)
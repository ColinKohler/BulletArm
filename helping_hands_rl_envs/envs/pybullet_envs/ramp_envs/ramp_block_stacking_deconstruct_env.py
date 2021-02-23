import numpy as np
from itertools import combinations
from helping_hands_rl_envs.envs.pybullet_envs.ramp_envs.ramp_deconstruct_env import RampDeconstructEnv
from helping_hands_rl_envs.envs.pybullet_envs.block_stacking_deconstruct_env import BlockStackingDeconstructEnv
from helping_hands_rl_envs.simulators import constants

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
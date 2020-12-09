from envs.pybullet_envs.ramp_envs.ramp_deconstruct_env import RampDeconstructEnv
from envs.pybullet_envs.improvise_house_building_2_deconstruct_env import ImproviseHouseBuilding2Env

class RampImproviseHouseBuilding2DeconstructEnv(RampDeconstructEnv, ImproviseHouseBuilding2Env):
  ''''''

  def __init__(self, config):
    RampDeconstructEnv.__init__(self, config)

  def generateStructure(self):
    ImproviseHouseBuilding2Env.generateStructure(self)

  def checkStructure(self):
    ''''''
    return ImproviseHouseBuilding2Env.checkStructure(self)

  def isSimValid(self):
    return RampDeconstructEnv.isSimValid(self)

def createRampImproviseHouseBuilding2DeconstructEnv(config):
  return RampImproviseHouseBuilding2DeconstructEnv(config)
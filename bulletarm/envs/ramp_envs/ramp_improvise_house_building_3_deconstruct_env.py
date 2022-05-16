from bulletarm.envs.ramp_envs.ramp_deconstruct_env import RampDeconstructEnv
from bulletarm.envs.deconstruct_envs.improvise_house_building_3_deconstruct_env import ImproviseHouseBuilding3DeconstructEnv


class RampImproviseHouseBuilding3DeconstructEnv(RampDeconstructEnv, ImproviseHouseBuilding3DeconstructEnv):
  ''''''
  def __init__(self, config):
    RampDeconstructEnv.__init__(self, config)

  def generateStructure(self):
    ImproviseHouseBuilding3DeconstructEnv.generateStructure(self)

  def checkStructure(self):
    ''''''
    return ImproviseHouseBuilding3DeconstructEnv.checkStructure(self)

  def isSimValid(self):
    return RampDeconstructEnv.isSimValid(self)

def createRampImproviseHouseBuilding3DeconstructEnv(config):
  return RampImproviseHouseBuilding3DeconstructEnv(config)

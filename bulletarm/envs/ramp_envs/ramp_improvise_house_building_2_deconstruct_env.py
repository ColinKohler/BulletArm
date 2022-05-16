from bulletarm.envs.ramp_envs.ramp_deconstruct_env import RampDeconstructEnv
from bulletarm.envs.deconstruct_envs.improvise_house_building_2_deconstruct_env import ImproviseHouseBuilding2DeconstructEnv

class RampImproviseHouseBuilding2DeconstructEnv(RampDeconstructEnv, ImproviseHouseBuilding2DeconstructEnv):
  ''''''

  def __init__(self, config):
    RampDeconstructEnv.__init__(self, config)

  def generateStructure(self):
    ImproviseHouseBuilding2DeconstructEnv.generateStructure(self)

  def checkStructure(self):
    ''''''
    return ImproviseHouseBuilding2DeconstructEnv.checkStructure(self)

  def isSimValid(self):
    return RampDeconstructEnv.isSimValid(self)

def createRampImproviseHouseBuilding2DeconstructEnv(config):
  return RampImproviseHouseBuilding2DeconstructEnv(config)
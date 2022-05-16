from bulletarm.envs.ramp_envs.ramp_deconstruct_env import RampDeconstructEnv
from bulletarm.envs.deconstruct_envs.house_building_1_deconstruct_env import HouseBuilding1DeconstructEnv

class RampHouseBuilding1DeconstructEnv(RampDeconstructEnv, HouseBuilding1DeconstructEnv):
  ''''''
  def __init__(self, config):
    RampDeconstructEnv.__init__(self, config)

  def generateStructure(self):
    HouseBuilding1DeconstructEnv.generateStructure(self)

  def checkStructure(self):
    ''''''
    return HouseBuilding1DeconstructEnv.checkStructure(self)

  def isSimValid(self):
    return RampDeconstructEnv.isSimValid(self)

def createRampHouseBuilding1DeconstructEnv(config):
  return RampHouseBuilding1DeconstructEnv(config)
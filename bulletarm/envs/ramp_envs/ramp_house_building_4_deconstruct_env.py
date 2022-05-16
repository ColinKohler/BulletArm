from bulletarm.envs.ramp_envs.ramp_deconstruct_env import RampDeconstructEnv
from bulletarm.envs.deconstruct_envs.house_building_4_deconstruct_env import HouseBuilding4DeconstructEnv

class RampHouseBuilding4DeconstructEnv(RampDeconstructEnv, HouseBuilding4DeconstructEnv):
  ''''''

  def __init__(self, config):
    RampDeconstructEnv.__init__(self, config)

  def generateStructure(self):
    HouseBuilding4DeconstructEnv.generateStructure(self)

  def checkStructure(self):
    ''''''
    return HouseBuilding4DeconstructEnv.checkStructure(self)

  def isSimValid(self):
    return RampDeconstructEnv.isSimValid(self)

def createRampHouseBuilding4DeconstructEnv(config):
  return RampHouseBuilding4DeconstructEnv(config)
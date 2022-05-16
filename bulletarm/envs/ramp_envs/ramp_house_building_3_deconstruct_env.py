from bulletarm.envs.ramp_envs.ramp_deconstruct_env import RampDeconstructEnv
from bulletarm.envs.deconstruct_envs.house_building_3_deconstruct_env import HouseBuilding3DeconstructEnv

class RampHouseBuilding3DeconstructEnv(RampDeconstructEnv, HouseBuilding3DeconstructEnv):
  ''''''

  def __init__(self, config):
    RampDeconstructEnv.__init__(self, config)

  def generateStructure(self):
    HouseBuilding3DeconstructEnv.generateStructure(self)

  def checkStructure(self):
    ''''''
    return HouseBuilding3DeconstructEnv.checkStructure(self)

  def isSimValid(self):
    return RampDeconstructEnv.isSimValid(self)

def createRampHouseBuilding3DeconstructEnv(config):
  return RampHouseBuilding3DeconstructEnv(config)
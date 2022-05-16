from bulletarm.envs.ramp_envs.ramp_deconstruct_env import RampDeconstructEnv
from bulletarm.envs.deconstruct_envs.house_building_2_deconstruct_env import HouseBuilding2DeconstructEnv

class RampHouseBuilding2DeconstructEnv(RampDeconstructEnv, HouseBuilding2DeconstructEnv):
  ''''''

  def __init__(self, config):
    RampDeconstructEnv.__init__(self, config)

  def generateStructure(self):
    HouseBuilding2DeconstructEnv.generateStructure(self)

  def checkStructure(self):
    ''''''
    return HouseBuilding2DeconstructEnv.checkStructure(self)

  def isSimValid(self):
    return RampDeconstructEnv.isSimValid(self)

def createRampHouseBuilding2DeconstructEnv(config):
  return RampHouseBuilding2DeconstructEnv(config)
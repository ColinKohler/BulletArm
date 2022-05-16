from bulletarm.envs.deconstruct_envs.deconstruct_env import DeconstructEnv
from bulletarm.pybullet.utils import constants

class HouseBuilding1DeconstructEnv(DeconstructEnv):
  ''''''
  def __init__(self, config):
    # env specific parameters
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.6, 0.6]
    if 'num_objects' not in config:
      config['num_objects'] = 4
    if 'max_steps' not in config:
      config['max_steps'] = 10
    super(HouseBuilding1DeconstructEnv, self).__init__(config)

  def checkStructure(self):
    ''''''
    blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
    triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))
    return self._checkStack(blocks+triangles) and self._checkObjUpright(triangles[0])

  def generateStructure(self):
    padding = self.max_block_size * 1.5
    pos = self.get1BaseXY(padding)
    rot = self._getValidOrientation(self.random_orientation)
    for i in range(self.num_obj - 1):
      self.generateStructureShape((pos[0], pos[1], i * self.max_block_size + self.max_block_size / 2), rot,
                                  constants.CUBE)
    self.generateStructureShape((pos[0], pos[1], (self.num_obj - 1) * self.max_block_size + self.max_block_size / 2),
                                rot, constants.TRIANGLE)
    self.wait(50)

  def isSimValid(self):
    triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))
    return self._checkObjUpright(triangles[0]) and DeconstructEnv.isSimValid(self)


def createHouseBuilding1DeconstructEnv(config):
  return HouseBuilding1DeconstructEnv(config)

import numpy as np

from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils import constants

class PyramidStackingEnv(BaseEnv):
  ''''''
  def __init__(self, config):
    # env specific parameters
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.6, 0.6]
    if 'num_objects' not in config:
      config['num_objects'] = 3
    if 'max_steps' not in config:
      config['max_steps'] = 10
    super(PyramidStackingEnv, self).__init__(config)

  def reset(self):
    ''''''
    self.resetPybulletWorkspace()
    self._generateShapes(constants.CUBE, self.num_obj, random_orientation=self.random_orientation)
    return self._getObservation()

  def _checkTermination(self):
    ''''''
    obj_z = [obj.getZPosition() for obj in self.objects]
    if np.allclose(obj_z[0], obj_z):
      return False

    top_obj = self.objects[np.argmax(obj_z)]
    mask = np.array([True] * self.num_obj)
    mask[np.argmax(obj_z)] = False
    bottom_objs = np.array(self.objects)[mask]
    return self._checkInBetween(top_obj, bottom_objs[0], bottom_objs[1], threshold=0.01) and \
           self._checkAdjacent(bottom_objs[0], bottom_objs[1])

def createPyramidStackingEnv(config):
  return PyramidStackingEnv(config)

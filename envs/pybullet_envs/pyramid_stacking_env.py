import numpy as np

from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants

class PyramidStackingEnv(PyBulletEnv):
  ''''''
  def __init__(self, config):
    super(PyramidStackingEnv, self).__init__(config)

  def reset(self):
    ''''''
    super(PyramidStackingEnv, self).reset()
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

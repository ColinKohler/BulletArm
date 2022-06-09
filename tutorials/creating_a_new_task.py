import numpy as np

from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils import constants

'''
The process of creating a new task can be summarized as follows:
  1.) Create a new env class which subclasses the BaseEnv
  2.) Define the reset() and _checkTermination() functions to specify the initial and goal states of the task
  3.) Define a createEnv() function which wraps the instantiation of the new class
  4.) Add your new class to the CREATE_ENV_FNS dict in bulletarm/envs/env_fn.py; this allows the env_factory to
      find your new task.

Below we show how to create a new task which defines a pyramid block stacking task.
'''

# New tasks must subclass the BaseEnv
class PyramidStackingEnv(BaseEnv):
  ''' The pyramid stacking task.

  The robot needs to stack 2 cubic blocks adjacient and then stack another cubic block on
  top and in-between the base blocks.

  Args:
    config (dict): Intialization arguments for the env
  '''
  def __init__(self, config):
    # Any task specific initialization can be done here but often it is not required as most
    # information will be contained within the configuration dictionary.
    super().__init__(config)

  # The two functions which MUST be defined for all tasks are the reset function and the _checkTermination function.
  # The reset function sepcifies the initial state for the task including the objects which will be manipulated.
  def reset(self):
    self.resetPybulletWorkspace()
    self._generateShapes(constants.CUBE, self.num_obj, random_orientation=self.random_orientation)
    return self._getObservation()

  # The _checkTermination function defines the goal state for the task.
  def _checkTermination(self):
    obj_z = [obj.getZPosition() for obj in self.objects]
    if np.allclose(obj_z[0], obj_z):
      return False

    top_obj = self.objects[np.argmax(obj_z)]
    mask = np.array([True] * self.num_obj)
    mask[np.argmax(obj_z)] = False
    bottom_objs = np.array(self.objects)[mask]
    return self._checkInBetween(top_obj, bottom_objs[0], bottom_objs[1], threshold=0.01) and \
           self._checkAdjacent(bottom_objs[0], bottom_objs[1])

# This helper function is required for the multi threading in order to run multiple envs in parallel.
def createPyramidStackingEnv(config):
  return PyramidStackingEnv(config)

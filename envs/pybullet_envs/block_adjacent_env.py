import numpy as np

from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants

class BlockAdjacentEnv(PyBulletEnv):
  ''''''
  def __init__(self, config):
    super(BlockAdjacentEnv, self).__init__(config)

  def reset(self):
    ''''''
    super(BlockAdjacentEnv, self).reset()
    self._generateShapes(constants.CUBE, self.num_obj, random_orientation=self.random_orientation)
    return self._getObservation()

  def _checkTermination(self):
    ''''''
    pos = np.array([o.getPosition() for o in self.objects])
    if (pos[:,2].max() - pos[:,2].min()) > 0.01: return False

    if np.allclose(pos[:,0], pos[0,0], atol=0.01):
      return np.abs(pos[:,1].max() - pos[:,1].min()) < self.max_block_size * 3.5
    elif np.allclose(pos[:,1], pos[0,1], atol=0.01):
      return np.abs(pos[:,0].max() - pos[:,0].min()) < self.max_block_size * 3.5
    else:
      return False

def createBlockAdjacentEnv(config):
  def _thunk():
    return BlockAdjacentEnv(config)
  return _thunk

import numpy as np

from helping_hands_rl_envs.envs.pybullet_envs.two_view_envs.drawer_teapot_env import DrawerTeapotEnv
from helping_hands_rl_envs.simulators.pybullet.utils.renderer import Renderer

class MViewDrawerTeapotEnv(DrawerTeapotEnv):
  def __init__(self, config):
    super().__init__(config)
    self.renderer = Renderer(self.workspace)
    self.view_thetas = [-np.pi / 9, 0, np.pi / 9]

  def _getObservation(self, action=None):
    ''''''
    # TODO:
    self.renderer.getNewPointCloud()
    topdown_heightmap = self.renderer.getTopDownHeightmap(self.heightmap_size)
    forward_heightmaps = self.renderer.getForwardHeightmapByThetas(self.heightmap_size, self.view_thetas)
    pos = self.drawer1.getHandlePosition()
    old_heightmap = self.heightmap
    self.heightmap = topdown_heightmap

    if action is None or self._isHolding() == False:
      in_hand_img = self.getEmptyInHand()
    else:
      motion_primative, x, y, z, rot = self._decodeAction(action)
      in_hand_img = self.getInHandImage(old_heightmap, x, y, z, rot, self.heightmap)

    # forward_heightmap = self._getHeightmapForward()
    heightmaps = np.concatenate((topdown_heightmap.reshape(1, *topdown_heightmap.shape), forward_heightmaps), 0)
    heightmaps = np.moveaxis(heightmaps, 0, -1)

    return self._isHolding(), in_hand_img, heightmaps

def createMViewDrawerTeapotEnv(config):
  return MViewDrawerTeapotEnv(config)

if __name__ == '__main__':
  workspace = np.asarray([[0.3, 0.7],
                          [-0.2, 0.2],
                          [0, 0.40]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyrr', 'num_objects': 5, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point', 'physics_mode': 'slow'}
  env = MViewDrawerTeapotEnv(env_config)
  while True:
    s, in_hand, obs = env.reset()

import numpy as np

from helping_hands_rl_envs.envs.pybullet_envs.household_envs.drawer_teapot_env import DrawerTeapotEnv
from helping_hands_rl_envs.simulators.pybullet.utils.renderer import Renderer
from helping_hands_rl_envs.simulators.pybullet.utils.sensor import Sensor

class MViewDrawerTeapotEnv(DrawerTeapotEnv):
  def __init__(self, config):
    super().__init__(config)
    self.view_type = config['view_type']
    self.view_thetas = [-np.pi / 9, 0, np.pi / 9]
    assert self.view_type in ['render', 'camera']
    if self.view_type == 'render':
      self.renderer = Renderer(self.workspace)
    else:
      self.forward_cams = []
      for theta in self.view_thetas:
        dy = np.sin(theta) * 20
        dx = np.cos(theta) * 20
        cam_target_pos = [self.workspace[0].mean() + 0.41, self.workspace[1].mean(), self.workspace[2].mean()]
        cam_up_vector = [0, 0, 1]
        cam_pos = [self.workspace[0].mean() + 0.41 - dx, -dy, self.workspace[2].mean()]
        cam = Sensor(cam_pos, cam_up_vector, cam_target_pos, self.workspace[2][1] - self.workspace[2][0], 20-(0.41+self.workspace[0].mean())/np.cos(theta), 20+self.wall_x)
        self.forward_cams.append(cam)

  def _getObservation(self, action=None):
    ''''''
    # TODO:
    if self.view_type == 'render':
      self.renderer.getNewPointCloud()
      topdown_heightmap = self.renderer.getTopDownHeightmap(self.heightmap_size)
      forward_heightmaps = self.renderer.getForwardHeightmapByThetas(self.heightmap_size, self.view_thetas)
    else:
      forward_heightmaps = [cam.getHeightmap(self.heightmap_size) for cam in self.forward_cams]
      forward_heightmaps = np.stack(forward_heightmaps)
      topdown_heightmap = self.sensor.getHeightmap(self.heightmap_size)
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

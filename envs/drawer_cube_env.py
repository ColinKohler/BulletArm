import pybullet as pb
import numpy as np

from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv
from envs.pybullet_envs.two_view_envs.two_view_drawer_env import TwoViewDrawerEnv
from helping_hands_rl_envs.simulators import constants

def createDrawerCubeEnv(simulator_base_env, config):
  class DrawerCubeEnv(TwoViewDrawerEnv):
    def __init__(self, config):
      if simulator_base_env is PyBulletEnv:
        super().__init__(config)
      else:
        raise ValueError('Bad simulator base env specified.')
      self.simulator_base_env = simulator_base_env
      self.random_orientation = config['random_orientation'] if 'random_orientation' in config else False
      self.num_obj = config['num_objects'] if 'num_objects' in config else 1
      self.reward_type = config['reward_type'] if 'reward_type' in config else 'sparse'
      self.handle_pos = None
      self.handle2_pos = None

    def reset(self):
      super().reset()
      if np.random.random() > 0.5:
        self._generateShapes(constants.CUBE, 1, pos=[[0.65, 0, 0.1]])
      else:
        self._generateShapes(constants.CUBE, 1, pos=[[0.65, 0, 0.3]])
      # self._generateShapes(constants.CUBE, 1, pos=[[0.65, 0, 0.1]])

      self.handle_pos = self.drawer1.getHandlePosition()
      self.handle2_pos = self.drawer2.getHandlePosition()
      return self._getObservation()

    def step(self, action):
      motion_primative, x, y, z, rot = self._decodeAction(action)
      self.takeAction(action)
      self.wait(100)
      obs = self._getObservation(action)
      done = self._checkTermination()
      reward = 1.0 if done else 0.0

      if not done:
        done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
      self.current_episode_steps += 1

      return obs, reward, done

    def _checkTermination(self):
      return self._isObjOnGround(self.objects[0]) and self._isPointInWorkspace(self.objects[0].getPosition())



    def test(self):
      handle1_pos = self.drawer1.getHandlePosition()
      handle2_pos = self.drawer2.getHandlePosition()
      rot = pb.getQuaternionFromEuler((0, -np.pi / 2, 0))
      self.robot.pull(handle1_pos, rot, 0.2)
      self.robot.pull(handle2_pos, rot, 0.2)

      self.robot.pick(self.objects[0].getPosition(), pb.getQuaternionFromEuler((0, 0, 0)), 0.1,  False, objects=self.objects)
      self.robot.place([0.3, 0, 0], pb.getQuaternionFromEuler((0, 0, 0)), 0.1, False)


  def _thunk():
    return DrawerCubeEnv(config)

  return _thunk

if __name__ == '__main__':
  workspace = np.asarray([[0.3, 0.7],
                          [-0.2, 0.2],
                          [0, 0.40]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyrr', 'num_objects': 5, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point'}
  env = createDrawerCubeEnv(PyBulletEnv, env_config)()
  while True:
    s, in_hand, obs = env.reset()
    env.step(3, )
    env.test()
    env.test()
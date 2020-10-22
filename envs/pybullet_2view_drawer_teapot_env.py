import pybullet as pb
import numpy as np

from helping_hands_rl_envs.simulators.pybullet.equipments.drawer import Drawer
from simulators.pybullet.objects.teapot_base import TeapotBase
from simulators.pybullet.objects.teapot_lid import TeapotLid
from helping_hands_rl_envs.envs.pybullet_2view_drawer_env import PyBullet2ViewDrawerEnv


class PyBullet2ViewDrawerTeapotEnv(PyBullet2ViewDrawerEnv):
  def __init__(self, config):
    super().__init__(config)

  def reset(self):
    super().reset()
    self.generateTeapot()
    return self._getObservation()

  def step(self, action):
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
    return self._isObjOnGround(self.objects[0]) and self._checkOnTop(self.objects[0], self.objects[1])

  def generateTeapot(self):
    rot = (np.random.random() - 0.5) * np.pi
    if np.random.random() > 0.5:
      teapot = TeapotBase([0.73, 0, 0.05], pb.getQuaternionFromEuler((0, 0, rot)), 0.1)
      teapot_lid = TeapotLid([0.73, 0, 0.25], pb.getQuaternionFromEuler((0, 0, 0)), 0.1)
    else:
      teapot = TeapotBase([0.73, 0, 0.25], pb.getQuaternionFromEuler((0, 0, rot)), 0.1)
      teapot_lid = TeapotLid([0.73, 0, 0.05], pb.getQuaternionFromEuler((0, 0, 0)), 0.1)
    self.objects.append(teapot)
    self.objects.append(teapot_lid)

  def test(self):
    for _ in range(100):
      pb.stepSimulation()
    handle1_pos = self.drawer.getHandlePosition()
    handle2_pos = self.drawer2.getHandlePosition()
    rot = pb.getQuaternionFromEuler((0, -np.pi/2, 0))
    self.robot.pull(handle1_pos, rot, 0.2)

    teapot_handle_pos = self.objects[0].getHandlePos()
    self.robot.pick(teapot_handle_pos, self.objects[0].getRotation(), 0.1, objects=self.objects, dynamic=False)
    self.robot.place([0.3, -0.15, 0.1], [0, 0, 0, 1], 0.1, False)

    self.robot.pull(handle2_pos, rot, 0.2)
    lid_pos = self.objects[1].getPosition()
    self.robot.pick(lid_pos, self.objects[1].getRotation(), 0.1, objects=self.objects, dynamic=False)
    teapot_open_pos = self.objects[0].getOpenPos()
    self.robot.place(teapot_open_pos, (0, 0, 0, 1), 0.1, False)
    pass


if __name__ == '__main__':
  workspace = np.asarray([[0.3, 0.7],
                          [-0.2, 0.2],
                          [0, 0.40]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyrr', 'num_objects': 5, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point'}
  env = PyBullet2ViewDrawerTeapotEnv(env_config)
  while True:
    s, in_hand, obs = env.reset()
    env.test()
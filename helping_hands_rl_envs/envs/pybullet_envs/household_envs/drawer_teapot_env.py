import pybullet as pb
import numpy as np

from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.objects.teapot_base import TeapotBase
from helping_hands_rl_envs.simulators.pybullet.objects.teapot_lid import TeapotLid
from helping_hands_rl_envs.envs.pybullet_envs.household_envs.drawer_env import DrawerEnv


class DrawerTeapotEnv(DrawerEnv):
  def __init__(self, config):
    super().__init__(config)

  def resetDrawerTeapotEnv(self):
    self.resetDrawerEnv()
    self.generateTeapot()

  def reset(self):
    self.resetDrawerTeapotEnv()
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
    teapot_model_id = np.random.choice([1, 2, 3, 4, 5])
    rot = (np.random.random() - 0.5) * np.pi + self.drawer_theta
    if np.random.random() > 0.5:
      teapot = TeapotBase(self.drawer1.getObjInitPos(), pb.getQuaternionFromEuler((0, 0, rot)), 0.08, teapot_model_id)
      teapot_lid = TeapotLid(self.drawer2.getObjInitPos(), pb.getQuaternionFromEuler((0, 0, rot)), 0.08, teapot_model_id)
    else:
      teapot = TeapotBase(self.drawer2.getObjInitPos(), pb.getQuaternionFromEuler((0, 0, rot)), 0.08, teapot_model_id)
      teapot_lid = TeapotLid(self.drawer1.getObjInitPos(), pb.getQuaternionFromEuler((0, 0, rot)), 0.08, teapot_model_id)
    self.objects.append(teapot)
    self.object_types[teapot] = constants.TEAPOT
    self.objects.append(teapot_lid)
    self.object_types[teapot_lid] = constants.TEAPOT_LID
    for _ in range(100):
      pb.stepSimulation()

  def test(self):
    for _ in range(100):
      pb.stepSimulation()
    a = self.drawer1.isObjInsideDrawer(self.objects[0])
    handle1_pos = self.drawer1.getHandlePosition()
    handle2_pos = self.drawer2.getHandlePosition()
    rot = pb.getQuaternionFromEuler((0, -np.pi/2, 0))
    self.robot.pull(handle1_pos, rot, 0.25, False)

    teapot_handle_pos = self.objects[0].getGraspPosition()
    self.robot.pick(teapot_handle_pos, self.objects[0].getRotation(), 0.1, objects=self.objects, dynamic=False)
    self.robot.place([0.37, 0, 0.1], [0, 0, 0, 1], 0.1, False)

    self.robot.pull(handle2_pos, rot, 0.25, False)
    lid_pos = self.objects[1].getPosition()
    self.robot.pick(lid_pos, self.objects[1].getRotation(), 0.1, objects=self.objects, dynamic=False)
    teapot_open_pos = self.objects[0].getOpenPos()
    self.robot.place(teapot_open_pos, (0, 0, 0, 1), 0.1, False)
    pass

def createDrawerTeapotEnv(config):
  return DrawerTeapotEnv(config)

if __name__ == '__main__':
  workspace = np.asarray([[0.3, 0.7],
                          [-0.2, 0.2],
                          [0, 0.40]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyrr', 'num_objects': 5, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point', 'physics_mode': 'slow'}
  env = DrawerTeapotEnv(env_config)
  while True:
    s, in_hand, obs = env.reset()
    env.test()
from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants

class RandomStackingEnv(PyBulletEnv):
  ''''''
  def __init__(self, config):
    config['check_random_obj_valid'] = True
    super(RandomStackingEnv, self).__init__(config)

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

  def reset(self):
    ''''''
    self.resetPybulletEnv()
    self._generateShapes(constants.RANDOM, self.num_obj, random_orientation=self.random_orientation, z_scale=1.5)
    return self._getObservation()

  def _checkTermination(self):
    ''''''
    return self._checkStack()

def createRandomStackingEnv(config):
  return RandomStackingEnv(config)

from copy import deepcopy
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants
from utils.check_goal import CheckGoal

def createHouseBuildingXEnv(simulator_base_env, config):
  class HouseBuildingXEnv(simulator_base_env):
    ''''''
    def __init__(self, config):
      if simulator_base_env is PyBulletEnv:
        super().__init__(config)
        self.block_scale_range = (0.6, 0.6)
        self.min_block_size = self.block_original_size * self.block_scale_range[0]
        self.max_block_size = self.block_original_size * self.block_scale_range[1]
      else:
        raise ValueError('Bad simulator base env specified.')
      self.simulator_base_env = simulator_base_env
      self.random_orientation = config['random_orientation'] if 'random_orientation' in config else False
      self.num_obj = config['num_objects'] if 'num_objects' in config else 1
      self.reward_type = config['reward_type'] if 'reward_type' in config else 'sparse'

      goal = config["goal_string"]
      self.check_goal = CheckGoal(goal, self)

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
      while True:
        super(HouseBuildingXEnv, self).reset()
        try:
          # order matters!
          self._generateShapes(constants.ROOF, 1, random_orientation=self.random_orientation)
          self._generateShapes(constants.BRICK, 2, random_orientation=self.random_orientation)
          self._generateShapes(constants.CUBE, 4, random_orientation=self.random_orientation)
          self._generateShapes(constants.TRIANGLE, 1, random_orientation=self.random_orientation)
        except Exception as e:
          continue
        else:
          break
      return self._getObservation()

    def _checkTermination(self):
      return self.check_goal.check()

    def getObjectPosition(self):
      return list(map(self._getObjectPosition, self.objects))

    def isSimValid(self):
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
      return self._checkObjUpright(roofs[0]) and super().isSimValid()

  def _thunk():
    return HouseBuildingXEnv(config)

  return _thunk
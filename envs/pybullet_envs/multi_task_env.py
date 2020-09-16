from helping_hands_rl_envs.envs.pybullet_envs.block_stacking_env import createBlockStackingEnv
from helping_hands_rl_envs.envs.pybullet_envs.brick_stacking_env import createBrickStackingEnv
from helping_hands_rl_envs.envs.pybullet_envs.block_adjacent_env import createBlockAdjacentEnv
from helping_hands_rl_envs.envs.pybullet_envs.house_building_1_env import createHouseBuilding1Env

def createMultiTaskEnv(configs):
  class MultiTaskEnv(object):
    ''''''
    def __init__(self, configs):
      self.envs = list()
      self.env_types = list()
      for config in configs:
        if config['env_type'] == 'block_stacking':
          self.envs.append(createBlockStackingEnv(config)())
          self.env_types.append('block_stacking')
        elif config['env_type'] == 'brick_stacking':
          self.envs.append(createBrickStackingEnv(config)())
          self.env_types.append('brick_stacking')
        elif config['env_type'] == 'block_adjacent':
          self.envs.append(createBlockAdjacentEnv(config)())
          self.env_types.append('block_adjacent')
        elif config['env_type'] == 'house_building_1':
          self.envs.append(createHouseBuilding1Env(config)())
          self.env_types.append('house_building_1')
        else:
          raise ValueError('Env type not implemented in Multi-task env.')

      self.active_env_id = 0
      self.active_env = self.envs[self.active_env_id]

    def step(self, action):
      return self.active_env.step(action)

    def reset(self):
      ''''''
      self.active_env_id = (self.active_env_id + 1) % len(self.envs)
      self.active_env = self.envs[self.active_env_id]

      return self.active_env.reset()

    def _checkTermination(self):
      ''''''
      return self.active_env._checkTermination()

    def isSimValid(self):
      return self.active_env.isSimValid()

    def didBlockFall(self):
      return self.active_env.didBlockFall()

  def _thunk():
    return MultiTaskEnv(configs)

  return _thunk

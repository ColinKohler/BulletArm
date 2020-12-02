from helping_hands_rl_envs.envs.block_stacking_env import createBlockStackingEnv
from helping_hands_rl_envs.envs.brick_stacking_env import createBrickStackingEnv
from helping_hands_rl_envs.envs.pyramid_stacking_env import createPyramidStackingEnv
from helping_hands_rl_envs.envs.block_adjacent_env import createBlockAdjacentEnv
from helping_hands_rl_envs.envs.house_building_1_env import createHouseBuilding1Env

def createMultiTaskEnv(simulator_base_env, configs):
  class MultiTaskEnv(object):
    ''''''
    def __init__(self, configs):
      self.simulator_base_env = simulator_base_env
      self.seed = configs[0]['seed']

      self.envs = list()
      self.env_types = list()
      for config in configs:
        if config['env_type'] == 'block_stacking':
          self.envs.append(createBlockStackingEnv(simulator_base_env, config)())
          self.env_types.append('block_stacking')
        elif config['env_type'] == 'brick_stacking':
          self.envs.append(createBrickStackingEnv(simulator_base_env, config)())
          self.env_types.append('brick_stacking')
        elif config['env_type'] == 'pyramid_stacking':
          self.envs.append(createPyramidStackingEnv(simulator_base_env, config)())
          self.env_types.append('pyramid_stacking')
        elif config['env_type'] == 'block_adjacent':
          self.envs.append(createBlockAdjacentEnv(simulator_base_env, config)())
          self.env_types.append('block_adjacent')
        elif config['env_type'] == 'house_building_1':
          self.envs.append(createHouseBuilding1Env(simulator_base_env, config)())
          self.env_types.append('house_building_1')
        else:
          raise ValueError('Env type not implemented in Multi-task env.')

      self.active_env_id = 0
      self.active_env = self.envs[self.active_env_id]

    def __getattr__(self, attr):
      return self.active_env.__getattribute__(attr)

    def reset(self):
      ''''''
      self.active_env_id = (self.active_env_id + 1) % len(self.envs)
      self.active_env = self.envs[self.active_env_id]
      self.workspace = self.active_env.workspace

      return self.active_env.reset()

  def _thunk():
    return MultiTaskEnv(configs)

  return _thunk

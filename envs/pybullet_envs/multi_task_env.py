from helping_hands_rl_envs.envs.pybullet_envs.block_stacking_env import createBlockStackingEnv
from helping_hands_rl_envs.envs.pybullet_envs.pyramid_stacking_env import createPyramidStackingEnv
from helping_hands_rl_envs.envs.pybullet_envs.brick_stacking_env import createBrickStackingEnv
from helping_hands_rl_envs.envs.pybullet_envs.block_adjacent_env import createBlockAdjacentEnv
from helping_hands_rl_envs.envs.pybullet_envs.house_building_1_env import createHouseBuilding1Env
from helping_hands_rl_envs.envs.pybullet_envs.house_building_2_env import createHouseBuilding2Env
from helping_hands_rl_envs.envs.pybullet_envs.house_building_3_env import createHouseBuilding3Env
from helping_hands_rl_envs.envs.pybullet_envs.house_building_4_env import createHouseBuilding4Env

class MultiTaskEnv(object):
  ''''''
  def __init__(self, configs):
    self.envs = list()
    self.env_types = list()
    for config in configs:
      if config['env_type'] == 'block_stacking':
        self.envs.append(createBlockStackingEnv(config))
        self.env_types.append('block_stacking')
      elif config['env_type'] == 'brick_stacking':
        self.envs.append(createBrickStackingEnv(config))
        self.env_types.append('brick_stacking')
      elif config['env_type'] == 'pyramid_stacking':
        self.envs.append(createPyramidStackingEnv(config))
        self.env_types.append('pyramid_stacking')
      elif config['env_type'] == 'block_adjacent':
        self.envs.append(createBlockAdjacentEnv(config))
        self.env_types.append('block_adjacent')
      elif config['env_type'] == 'house_building_1':
        self.envs.append(createHouseBuilding1Env(config))
        self.env_types.append('house_building_1')
      elif config['env_type'] == 'house_building_2':
        self.envs.append(createHouseBuilding2Env(config))
        self.env_types.append('house_building_2')
      elif config['env_type'] == 'house_building_3':
        self.envs.append(createHouseBuilding3Env(config))
        self.env_types.append('house_building_3')
      elif config['env_type'] == 'house_building_4':
        self.envs.append(createHouseBuilding4Env(config))
        self.env_types.append('house_building_4')
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

def createMultiTaskEnv(configs):
  return MultiTaskEnv(configs)

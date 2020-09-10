import copy
import numpy as np
import numpy.random as npr

from helping_hands_rl_envs.envs.numpy_env import NumpyEnv
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.envs.block_picking_env import createBlockPickingEnv
from helping_hands_rl_envs.envs.block_stacking_env import createBlockStackingEnv
from helping_hands_rl_envs.envs.brick_stacking_env import createBrickStackingEnv
from helping_hands_rl_envs.envs.block_cylinder_stacking_env import createBlockCylinderStackingEnv
from helping_hands_rl_envs.envs.house_building_1_env import createHouseBuilding1Env
from helping_hands_rl_envs.envs.house_building_2_env import createHouseBuilding2Env
from helping_hands_rl_envs.envs.house_building_3_env import createHouseBuilding3Env
from helping_hands_rl_envs.envs.house_building_4_env import createHouseBuilding4Env
from helping_hands_rl_envs.envs.house_building_5_env import createHouseBuilding5Env
from helping_hands_rl_envs.envs.house_building_x_env import createHouseBuildingXEnv
from helping_hands_rl_envs.envs.improvise_house_building_2_env import createImproviseHouseBuilding2Env
from helping_hands_rl_envs.envs.improvise_house_building_3_env import createImproviseHouseBuilding3Env
from helping_hands_rl_envs.envs.improvise_house_building_4_env import createImproviseHouseBuilding4Env
from helping_hands_rl_envs.envs.house_building_1_deconstruct_env import createHouseBuilding1DeconstructEnv
from helping_hands_rl_envs.envs.house_building_4_deconstruct_env import createHouseBuilding4DeconstructEnv
from helping_hands_rl_envs.envs.house_building_x_deconstruct_env import createHouseBuildingXDeconstructEnv
from helping_hands_rl_envs.envs.improvise_house_building_3_deconstruct_env import createImproviseHouseBuilding3DeconstructEnv
from helping_hands_rl_envs.envs.improvise_house_building_4_deconstruct_env import createImproviseHouseBuilding4DeconstructEnv
from helping_hands_rl_envs.envs.random_picking_env import createRandomPickingEnv
from helping_hands_rl_envs.envs.random_stacking_env import createRandomStackingEnv

from helping_hands_rl_envs.planners.planner_factory import createPlanner, AVAILABLE_PLANNER

from helping_hands_rl_envs.rl_runner import RLRunner
from helping_hands_rl_envs.data_runner import DataRunner

def createEnvs(num_processes, runner_type, simulator, env_type, env_config, planner_config={}):
  '''
  Create a number of environments on different processes to run in parralel

  Args:
    - num_processes: Number of envs to create
    - runner_type: data or rl runner
    - simulator: String indicating the type of simulator to use
    - env_type: String indicating the type of environment to create
    - conifg: Dict containing intialization arguments for the env

  Returns: EnvRunner containing all environments
  '''
  if 'action_sequence' not in env_config:
    env_config['action_sequence'] = 'pxyr'
  if 'simulate_grasp' not in env_config:
    env_config['simulate_grasp'] = True

  # Clone env config and generate random seeds for the different processes
  env_configs = [copy.copy(env_config) for _ in range(num_processes)]
  for i, env_config in enumerate(env_configs):
    env_config['seed'] = env_config['seed'] + i if 'seed' in env_config else npr.randint(100)

  # Set the super environment and add details to the configs as needed
  if simulator == 'pybullet':
    parent_env = PyBulletEnv
  elif simulator == 'numpy':
    parent_env = NumpyEnv
  else:
    raise ValueError('Invalid simulator passed to factory. Valid simulators are: \'numpy\', \'pybullet\'.')

  # Create the various environments
  if env_type == 'block_picking':
    envs = [createBlockPickingEnv(parent_env, env_configs[i]) for i in range(num_processes)]
  elif env_type == 'block_stacking':
    envs = [createBlockStackingEnv(parent_env, env_configs[i]) for i in range(num_processes)]
  elif env_type == 'brick_stacking':
    envs = [createBrickStackingEnv(parent_env, env_configs[i]) for i in range(num_processes)]
  elif env_type == 'block_cylinder_stacking':
    envs = [createBlockCylinderStackingEnv(parent_env, env_configs[i]) for i in range(num_processes)]
  elif env_type == 'house_building_1':
    envs = [createHouseBuilding1Env(parent_env, env_configs[i]) for i in range(num_processes)]
  elif env_type == 'house_building_2':
    envs = [createHouseBuilding2Env(parent_env, env_configs[i]) for i in range(num_processes)]
  elif env_type == 'house_building_3':
    envs = [createHouseBuilding3Env(parent_env, env_configs[i]) for i in range(num_processes)]
  elif env_type == 'house_building_4':
    envs = [createHouseBuilding4Env(parent_env, env_configs[i]) for i in range(num_processes)]
  elif env_type == 'house_building_5':
    envs = [createHouseBuilding5Env(parent_env, env_configs[i]) for i in range(num_processes)]
  elif env_type == 'house_building_x':
    envs = [createHouseBuildingXEnv(parent_env, env_configs[i]) for i in range(num_processes)]
  elif env_type == 'improvise_house_building_2':
    envs = [createImproviseHouseBuilding2Env(parent_env, env_configs[i]) for i in range(num_processes)]
  elif env_type == 'improvise_house_building_3':
    envs = [createImproviseHouseBuilding3Env(parent_env, env_configs[i]) for i in range(num_processes)]
  elif env_type == 'improvise_house_building_4':
    envs = [createImproviseHouseBuilding4Env(parent_env, env_configs[i]) for i in range(num_processes)]
  elif env_type == 'house_building_1_deconstruct':
    envs = [createHouseBuilding1DeconstructEnv(parent_env, env_configs[i]) for i in range(num_processes)]
  elif env_type == 'house_building_4_deconstruct':
    envs = [createHouseBuilding4DeconstructEnv(parent_env, env_configs[i]) for i in range(num_processes)]
  elif env_type == 'house_building_x_deconstruct':
    envs = [createHouseBuildingXDeconstructEnv(parent_env, env_configs[i]) for i in range(num_processes)]
  elif env_type == 'improvise_house_building_3_deconstruct':
    envs = [createImproviseHouseBuilding3DeconstructEnv(parent_env, env_configs[i]) for i in range(num_processes)]
  elif env_type == 'improvise_house_building_4_deconstruct':
    envs = [createImproviseHouseBuilding4DeconstructEnv(parent_env, env_configs[i]) for i in range(num_processes)]
  elif env_type == 'random_picking':
    envs = [createRandomPickingEnv(parent_env, env_configs[i]) for i in range(num_processes)]
  elif env_type == 'random_stacking':
    envs = [createRandomStackingEnv(parent_env, env_configs[i]) for i in range(num_processes)]
  else:
    raise ValueError('Invalid environment type passed to factory.')

  if 'planner' not in planner_config:
    if env_type in AVAILABLE_PLANNER:
      planner_config['planner'] = env_type
    else:
      planner_config['planner'] = 'random'

  planners = [createPlanner(planner_config) for i in range(num_processes)]
  if runner_type == 'rl':
    envs = RLRunner(envs, planners)
  elif runner_type == 'data':
    envs = DataRunner(envs, planners)
  else:
    raise ValueError('Invalid env runner type given. Must specify \'rl\', or \'data\'')

  return envs

import copy
import numpy as np
import numpy.random as npr

from helping_hands_rl_envs.envs.numpy_env import NumpyEnv
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.envs.block_picking_env import createBlockPickingEnv
from helping_hands_rl_envs.envs.block_stacking_env import createBlockStackingEnv
from helping_hands_rl_envs.envs.block_adjacent_env import createBlockAdjacentEnv
from helping_hands_rl_envs.envs.brick_stacking_env import createBrickStackingEnv
from helping_hands_rl_envs.envs.pyramid_stacking_env import createPyramidStackingEnv
from helping_hands_rl_envs.envs.block_cylinder_stacking_env import createBlockCylinderStackingEnv
from helping_hands_rl_envs.envs.house_building_1_env import createHouseBuilding1Env
from helping_hands_rl_envs.envs.house_building_2_env import createHouseBuilding2Env
from helping_hands_rl_envs.envs.house_building_3_env import createHouseBuilding3Env
from helping_hands_rl_envs.envs.multi_task_env import createMultiTaskEnv
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
  # TODO: This should also be cleaned up so env config is always the same
  if isinstance(env_config, list):
    for config in env_config:
      if 'action_sequence' not in config:
        config['action_sequence'] = 'pxyr'
      if 'simulate_grasp' not in config:
        config['simulate_grasp'] = True
  else:
    if 'action_sequence' not in env_config:
      env_config['action_sequence'] = 'pxyr'
    if 'simulate_grasp' not in env_config:
      env_config['simulate_grasp'] = True

  # Clone env config and generate random seeds for the different processes
  env_configs = [copy.deepcopy(env_config) for _ in range(num_processes)]
  for i, env_config in enumerate(env_configs):
    if isinstance(env_config, list): # Multi task
      for config in env_config:
        config['seed'] = config['seed'] + i if 'seed' in config else npr.randint(1000)
    else:
      env_config['seed'] = env_config['seed'] + i if 'seed' in env_config else npr.randint(1000)

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
  elif env_type == 'block_adjacent':
    envs = [createBlockAdjacentEnv(parent_env, env_configs[i]) for i in range(num_processes)]
  elif env_type == 'brick_stacking':
    envs = [createBrickStackingEnv(parent_env, env_configs[i]) for i in range(num_processes)]
  elif env_type == 'pyramid_stacking':
    envs = [createPyramidStackingEnv(parent_env, env_configs[i]) for i in range(num_processes)]
  elif env_type == 'block_cylinder_stacking':
    envs = [createBlockCylinderStackingEnv(parent_env, env_configs[i]) for i in range(num_processes)]
  elif env_type == 'house_building_1':
    envs = [createHouseBuilding1Env(parent_env, env_configs[i]) for i in range(num_processes)]
  elif env_type == 'house_building_2':
    envs = [createHouseBuilding2Env(parent_env, env_configs[i]) for i in range(num_processes)]
  elif env_type == 'house_building_3':
    envs = [createHouseBuilding3Env(parent_env, env_configs[i]) for i in range(num_processes)]
  elif env_type == 'multi_task':
    envs = [createMultiTaskEnv(parent_env, env_configs[i]) for i in range(num_processes)]
  else:
    raise ValueError('Invalid environment type passed to factory.')

  if 'planner_type' in planner_config:
    planner_type = planner_config['planner_type']
  elif env_type in AVAILABLE_PLANNER:
    planner_type = env_type
  else:
    planner_type = 'random'

  planners = [createPlanner(planner_type, planner_config) for i in range(num_processes)]
  if runner_type == 'rl':
    envs = RLRunner(envs, planners)
  elif runner_type == 'data':
    envs = DataRunner(envs, planners)
  else:
    raise ValueError('Invalid env runner type given. Must specify \'rl\', or \'data\'')

  return envs

import copy
import numpy as np
import numpy.random as npr

from helping_hands_rl_envs.envs.vrep_env import VrepEnv
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.envs.block_picking_env import createBlockPickingEnv
from helping_hands_rl_envs.envs.block_stacking_env import createBlockStackingEnv
from helping_hands_rl_envs.env_runner import EnvRunner

def createEnvs(num_processes, simulator, env_type, config):
  '''
  Create a number of environments on different processes to run in parralel

  Args:
    - num_processes: Number of envs to create
    - simulator: String indicating the type of simulator to use
    - env_type: String indicating the type of environment to create
    - conifg: Dict containing intialization arguments for the env

  Returns: EnvRunner containing all environments
  '''
  # Clone env config and generate random seeds for the different processes
  configs = [copy.copy(config) for _ in range(num_processes)]
  for config in configs:
    config['seed'] = npr.randint(100)

  # Set the super environment and add details to the configs as needed
  if simulator == 'vrep':
    for i in range(num_processes):
      configs[i]['port'] = configs[i]['port'] + i if configs[i]['port'] else 19997 + i
    parent_env = VrepEnv
  elif simulator == 'pybullet':
    parent_env = PyBulletEnv
  else:
    raise ValueError('Invalid simulator passed to factory. Valid simulators are: \'vrep\', \'pybullet\'.')

  # Create the various environments
  if env_type == 'block_picking':
    envs = [createBlockPickingEnv(parent_env, configs[i]) for i in range(num_processes)]
  elif env_type == 'block_stacking':
    envs = [createBlockStackingEnv(parent_env, configs[i]) for i in range(num_processes)]
  else:
    raise ValueError('Invalid environment type passed to factory. Valid types are: \'block_picking\', \'block_stacking\'.')

  envs = EnvRunner(envs)
  return envs

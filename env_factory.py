import numpy as np
import numpy.random as npr

from envs.vrep_env import VrepEnv
from envs.block_picking_env import BlockPickingEnv
from envs.block_stacking_env import BlockStackingEnv
from .env_runner import EnvRunner

def createEnvs(num_pocesses, simulator, env_type, config):
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
  configs = [config] * num_processes
  [config['seed'] = npr.randint(100) for _ in range(num_processes)]

  # Set the super environment and add details to the configs as needed
  if simulator == 'vrep':
    [config['port'] = 19997+i for i in range(num_processes)]
    parent_env = VrepEnv
  elif simulator == 'pybullet':
    parent_env = PyBulletEnv
  else:
    raise ValueError('Invalid simulator passed to factory. Valid simulators are: \'vrep\', \'pybullet\'.')

  # Create the various environments
  if env_type == 'block_picking':
    envs = [createBlockPickingEnv(parent_env, config) for i in range(num_processes)]
  elif env_type == 'block_stacking':
    envs = [createBlockStackingEnv(parent_env, config) for i in range(num_processes)]
  else:
    raise ValueError('Invalid environment type passed to factory. Valid types are: \'block_picking\', \'block_stacking\'.')

  envs = EnvRunner(envs)
  return envs

import copy
import numpy as np
import numpy.random as npr

from helping_hands_rl_envs.envs import env_constants
from helping_hands_rl_envs.envs.numpy_envs.numpy_env import NumpyEnv
from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.planners.planner_factory import getPlannerFn

from helping_hands_rl_envs.runner import MultiRunner, SingleRunner

def getEnvFn(simulator, env_type):
  '''
  Get the env creation function of the correct simulator and type.

  Args:
    -
    -

  Returns:
  '''
  if simulator == 'numpy':
    if env_type in env_constants.CREATE_NUMPY_ENV_FNS:
      return env_constants.CREATE_NUMPY_ENV_FNS[env_type]
    else:
      raise ValueError('Invalid environment type passed to factory. No numpy env for {}'.format(env_type))
  elif simulator == 'pybullet':
    if env_type in env_constants.CREATE_PYBULLET_ENV_FNS:
      return env_constants.CREATE_PYBULLET_ENV_FNS[env_type]
    else:
      raise ValueError('Invalid environment type passed to factory. No pybullet env for {}'.format(env_type))
  else:
    raise ValueError('Invalid simulator passed to factory. Valid simulators are: \'numpy\', \'pybullet\'.')

def createEnvs(num_processes, simulator, env_type, env_config, planner_config={}):
  '''
  Wrapper function to create either a single env the the main process or some
  number of envs each in their own seperate process.
  '''
  if num_processes == 0:
    return createSingleProcessEnv(simulator, env_type, env_config, planner_config)
  else:
    return createMultiprocessEnvs(num_processes, simulator, env_type, env_config, planner_config)

def createSingleProcessEnv(simulator, env_type, env_config, planner_config={}):
  '''
  Create a single environment

  Args:
    - simulator: String indicating the type of simulator to use
    - env_type: String indicating the type of environment to create
    - env_config: Dict containing intialization arguments for the env
    - planner_config: Dict containing intialization arguments for the planner

  Returns: SingleRunner containing the environment
  '''
  # Check to make sure a seed is given and generate a random one if it is not
  if env_type == 'multi_task':
    for config in env_config:
      config['seed'] = config['seed'] if 'seed' in config else npr.randint(1000)
  else:
    env_config['seed'] = env_config['seed'] if 'seed' in env_config else npr.randint(1000)

  # Create the environment and planner if planner_config exists
  env_func = getEnvFn(simulator, env_type)
  env = env_func(env_config)()

  if planner_config:
    planner = getPlannerFn(env_type, planner_config)(env)
  else:
    planner = None

  return SingleRunner(env, planner)

def createMultiprocessEnvs(num_processes, simulator, env_type, env_config, planner_config={}):
  '''
  Create a number of environments on different processes to run in parralel

  Args:
    - num_processes: Number of envs to create
    - simulator: String indicating the type of simulator to use
    - env_type: String indicating the type of environment to create
    - env_config: Dict containing intialization arguments for the env
    - planner_config: Dict containing intialization arguments for the planner

  Returns: MultiRunner containing all environments
  '''
  # Clone env config and set seeds for the different processes
  env_configs = [copy.deepcopy(env_config) for _ in range(num_processes)]
  for i, env_config in enumerate(env_configs):
    if env_type == 'multi_task':
      for config in env_config:
        config['seed'] = config['seed'] + i if 'seed' in config else npr.randint(1000)
    else:
      env_config['seed'] = env_config['seed'] + i if 'seed' in env_config else npr.randint(1000)

  # Create the various environments
  env_func = getEnvFn(simulator, env_type)
  envs = [env_func(env_configs[i]) for i in range(num_processes)]
  if planner_config:
    planners = [getPlannerFn(env_type, planner_config) for i in range(num_processes)]
  else:
    # TODO: This is a bit lazy and should probably just be a single None but that means
    #       more refactoring of the multi process stuff then I want to do atm
    planners = [None for i in range(num_processes)]

  return MultiRunner(envs, planners)

'''
.. moduleauthor: Colin Kohler <github.com/ColinKohler>
'''

import copy
import numpy as np
import numpy.random as npr

from bulletarm.envs import env_fn
from bulletarm.planners.planner_factory import getPlannerFn

from bulletarm.runner import MultiRunner, SingleRunner

def getEnvFn(env_type):
  '''
  Get the env creation function of the given type.

  Args:
    env_type (str): The type of environment to create

  Returns:
    function: A function which creates the env when run
  '''
  if env_type in env_fn.CREATE_ENV_FNS:
    return env_fn.CREATE_ENV_FNS[env_type]
  else:
    raise ValueError('Invalid environment type passed to factory. No env for {}'.format(env_type))

def createEnvs(num_processes, env_type, env_config={}, planner_config={}):
  '''
  Wrapper function to create either a single env the the main process or some
  number of envs each in their own seperate process.

  Args:
    num_processes (int): Number of envs to create
    env_type (str): The type of environment to create
    env_config (dict): Intialization arguments for the env
    planner_config (dict): Intialization arguments for the planner

  Returns:
    EnvRunner: SingleRunner or MultiRunner containing the environment
  '''
  if num_processes == 0:
    return createSingleProcessEnv(env_type, env_config, planner_config)
  else:
    return createMultiprocessEnvs(num_processes, env_type, env_config, planner_config)

def createSingleProcessEnv(env_type, env_config={}, planner_config={}):
  '''
  Create a single environment

  Args:
    env_type (str): The type of environment to create
    env_config (dict): Intialization arguments for the env
    planner_config (dict): Intialization arguments for the planner

  Returns:
    SingleRunner: SingleRunner containing the environment
  '''
  # Check to make sure a seed is given and generate a random one if it is not
  if env_type == 'multi_task':
    for config in env_config:
      config['seed'] = config['seed'] if 'seed' in config else npr.randint(1000)
  else:
    env_config['seed'] = env_config['seed'] if 'seed' in env_config else npr.randint(1000)

  # Create the environment and planner if planner_config exists
  env_func = getEnvFn(env_type)
  env = env_func(env_config)
  planner = getPlannerFn(env_type, planner_config)(env)
  return SingleRunner(env, planner)

def createMultiprocessEnvs(num_processes, env_type, env_config={}, planner_config={}):
  '''
  Create a number of environments on different processes to run in parralel

  Args:
    num_processes (int): Number of envs to create
    env_type (str): The type of environment to create
    env_config (dict): Intialization arguments for the env
    planner_config (dict): Intialization arguments for the planner

  Returns:
    MultiRunner: MultiRunner containing all environments
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
  env_func = getEnvFn(env_type)
  def getEnv(c):
    def _thunk():
      return env_func(c)
    return _thunk
  envs = [getEnv(env_configs[i]) for i in range(num_processes)]
  planners = [getPlannerFn(env_type, planner_config) for i in range(num_processes)]
  return MultiRunner(envs, planners)

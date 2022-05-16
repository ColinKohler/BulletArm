from bulletarm.planners import constants

def getPlannerFn(env_type, planner_config):
  '''

  '''
  if 'planner_type' in planner_config:
    planner_type = planner_config['planner_type']
  elif env_type in  constants.PLANNERS:
    planner_type = env_type
  else:
    planner_type = 'none'

  if planner_type in constants.PLANNERS:
    return lambda env: constants.PLANNERS[planner_type](env, planner_config)
  else:
    raise ValueError('Invalid planner passed to factory.')

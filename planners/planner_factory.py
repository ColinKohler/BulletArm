from helping_hands_rl_envs.planners.random_planner import RandomPlanner

def createPlanner(planner_type):
  if planner_type == 'random':
    return lambda env: RandomPlanner(env)

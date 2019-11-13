from helping_hands_rl_envs.planners.random_planner import RandomPlanner
from helping_hands_rl_envs.planners.block_picking_planner import BlockPickingPlanner
from helping_hands_rl_envs.planners.block_stacking_planner import BlockStackingPlanner

def createPlanner(planner_type):
  if planner_type == 'random':
    return lambda env: RandomPlanner(env)
  elif planner_type == 'block_picking':
    return lambda env: BlockPickingPlanner(env)
  elif planner_type == 'block_stacking':
    return lambda env: BlockStackingPlanner(env)

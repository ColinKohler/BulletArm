from helping_hands_rl_envs.planners.random_planner import RandomPlanner
from helping_hands_rl_envs.planners.play_planner import PlayPlanner
from helping_hands_rl_envs.planners.block_picking_planner import BlockPickingPlanner
from helping_hands_rl_envs.planners.block_stacking_planner import BlockStackingPlanner
from helping_hands_rl_envs.planners.brick_stacking_planner import BrickStackingPlanner

def createPlanner(config):
  if 'planner_noise' not in config: config['planner_noise'] = None

  if config['planner'] == 'random':
    return lambda env: RandomPlanner(env, config)
  if config['planner'] == 'play':
    return lambda env: PlayPlanner(env, config)
  elif config['planner'] == 'block_picking':
    return lambda env: BlockPickingPlanner(env, config)
  elif config['planner'] == 'block_stacking':
    return lambda env: BlockStackingPlanner(env, config)
  elif config['planner'] == 'brick_stacking':
    return lambda env: BrickStackingPlanner(env, config)

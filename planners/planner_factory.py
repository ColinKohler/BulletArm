from helping_hands_rl_envs.planners.random_planner import RandomPlanner
from helping_hands_rl_envs.planners.multi_task_planner import MultiTaskPlanner
from helping_hands_rl_envs.planners.play_planner import PlayPlanner
from helping_hands_rl_envs.planners.block_picking_planner import BlockPickingPlanner
from helping_hands_rl_envs.planners.block_stacking_planner import BlockStackingPlanner
from helping_hands_rl_envs.planners.block_adjacent_planner import BlockAdjacentPlanner
from helping_hands_rl_envs.planners.pyramid_stacking_planner import PyramidStackingPlanner
from helping_hands_rl_envs.planners.brick_stacking_planner import BrickStackingPlanner
from helping_hands_rl_envs.planners.house_building_1_planner import HouseBuilding1Planner
from helping_hands_rl_envs.planners.house_building_2_planner import HouseBuilding2Planner
from helping_hands_rl_envs.planners.house_building_3_planner import HouseBuilding3Planner

AVAILABLE_PLANNER = ['random',
                     'multi_task',
                     'play',
                     'block_picking',
                     'block_stacking',
                     'block_adjacent',
                     'pyramid_stacking',
                     'brick_stacking',
                     'house_building_1',
                     'house_building_2',
                     'house_building_3']

def createPlanner(planner, config):
  if isinstance(config, list):
    for c in config:
      if 'planner_noise' not in c: c['planner_noise'] = None
  else:
    if 'planner_noise' not in config: config['planner_noise'] = None

  if planner == 'random':
    return lambda env: RandomPlanner(env, config)
  elif planner == 'multi_task':
    return lambda env: MultiTaskPlanner(env, config)
  elif planner == 'play':
    return lambda env: PlayPlanner(env, config)
  elif planner == 'block_picking':
    return lambda env: BlockPickingPlanner(env, config)
  elif planner == 'block_stacking':
    return lambda env: BlockStackingPlanner(env, config)
  elif planner == 'block_adjacent':
    return lambda env: BlockAdjacentPlanner(env, config)
  elif planner == 'pyramid_stacking':
    return lambda env: PyramidStackingPlanner(env, config)
  elif planner == 'brick_stacking':
    return lambda env: BrickStackingPlanner(env, config)
  elif planner == 'house_building_1':
    return lambda env: HouseBuilding1Planner(env, config)
  elif planner == 'house_building_2':
    return lambda env: HouseBuilding2Planner(env, config)
  elif planner == 'house_building_3':
    return lambda env: HouseBuilding3Planner(env, config)

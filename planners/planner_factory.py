from helping_hands_rl_envs.planners.random_planner import RandomPlanner
from helping_hands_rl_envs.planners.play_planner import PlayPlanner
from helping_hands_rl_envs.planners.block_picking_planner import BlockPickingPlanner
from helping_hands_rl_envs.planners.block_stacking_planner import BlockStackingPlanner
from helping_hands_rl_envs.planners.brick_stacking_planner import BrickStackingPlanner
from helping_hands_rl_envs.planners.house_building_1_planner import HouseBuilding1Planner
from helping_hands_rl_envs.planners.house_building_2_planner import HouseBuilding2Planner
from helping_hands_rl_envs.planners.house_building_3_planner import HouseBuilding3Planner
from helping_hands_rl_envs.planners.house_building_4_planner import HouseBuilding4Planner
from helping_hands_rl_envs.planners.improvise_house_building_2_planner import ImproviseHouseBuilding2Planner
from helping_hands_rl_envs.planners.improvise_house_building_3_planner import ImproviseHouseBuilding3Planner
from helping_hands_rl_envs.planners.deconstruct_planner import DeconstructPlanner

AVAILABLE_PLANNER = ['random',
                     'play',
                     'block_picking',
                     'block_stacking',
                     'brick_stacking',
                     'house_building_1',
                     'house_building_2',
                     'house_building_3',
                     'house_building_4',
                     'improvise_house_building_2',
                     'improvise_house_building_3',
                     'house_building_1_deconstruct',
                     'house_building_4_deconstruct',
                     'house_building_x_deconstruct',
                     'improvise_house_building_3_deconstruct',
                     'improvise_house_building_4_deconstruct',
                     'random_picking',
                     'random_stacking']

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
  elif config['planner'] == 'house_building_1':
    return lambda env: HouseBuilding1Planner(env, config)
  elif config['planner'] == 'house_building_2':
    return lambda env: HouseBuilding2Planner(env, config)
  elif config['planner'] == 'house_building_3':
    return lambda env: HouseBuilding3Planner(env, config)
  elif config['planner'] == 'house_building_4':
    return lambda env: HouseBuilding4Planner(env, config)
  elif config['planner'] == 'improvise_house_building_2':
    return lambda env: ImproviseHouseBuilding2Planner(env, config)
  elif config['planner'] == 'improvise_house_building_3':
    return lambda env: ImproviseHouseBuilding3Planner(env, config)
  elif config['planner'] == 'house_building_1_deconstruct':
    return lambda env: DeconstructPlanner(env, config)
  elif config['planner'] == 'house_building_4_deconstruct':
    return lambda env: DeconstructPlanner(env, config)
  elif config['planner'] == 'house_building_x_deconstruct':
    return lambda env: DeconstructPlanner(env, config)
  elif config['planner'] == 'improvise_house_building_3_deconstruct':
    return lambda env: DeconstructPlanner(env, config)
  elif config['planner'] == 'improvise_house_building_4_deconstruct':
    return lambda env: DeconstructPlanner(env, config)
  elif config['planner'] == 'random_picking':
    return lambda env: BlockPickingPlanner(env, config)
  elif config['planner'] == 'random_stacking':
    return lambda env: BlockStackingPlanner(env, config)

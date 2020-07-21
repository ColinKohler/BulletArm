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
from helping_hands_rl_envs.planners.float_picking_planner import FloatPickingPlanner
from helping_hands_rl_envs.planners.block_placing_planner import BlockPlacingPlanner
from helping_hands_rl_envs.planners.tilt_block_stacking_planner import TiltBlockStackingPlanner
from helping_hands_rl_envs.planners.tilt_deconstruct_planner import TiltDeconstructPlanner

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
                     'improvise_house_building_3_deconstruct',
                     'improvise_house_building_4_deconstruct',
                     'random_picking',
                     'random_stacking',
                     'random_float_picking',
                     'cube_float_picking',
                     'block_placing',
                     'tilt_block_stacking',
                     'tilt_block_stacking_deconstruct',
                     'tilt_house_building_1_deconstruct',
                     'tilt_house_building_3_deconstruct',
                     'tilt_house_building_4_deconstruct',
                     'tilt_improvise_house_building_2_deconstruct',
                     'tilt_improvise_house_building_3_deconstruct',
                     'tilt_improvise_house_building_5_deconstruct',
                     'tilt_improvise_house_building_6_deconstruct',
                     ]

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
  elif config['planner'] == 'improvise_house_building_3_deconstruct':
    return lambda env: DeconstructPlanner(env, config)
  elif config['planner'] == 'improvise_house_building_4_deconstruct':
    return lambda env: DeconstructPlanner(env, config)
  elif config['planner'] == 'random_picking':
    return lambda env: BlockPickingPlanner(env, config)
  elif config['planner'] == 'random_stacking':
    return lambda env: BlockStackingPlanner(env, config)
  elif config['planner'] == 'random_float_picking':
    return lambda env: FloatPickingPlanner(env, config)
  elif config['planner'] == 'cube_float_picking':
    return lambda env: FloatPickingPlanner(env, config)
  elif config['planner'] == 'block_placing':
    return lambda env: BlockPlacingPlanner(env, config)
  elif config['planner'] == 'tilt_block_stacking':
    return lambda env: TiltBlockStackingPlanner(env, config)
  elif config['planner'] == 'tilt_block_stacking_deconstruct':
    return lambda env: TiltDeconstructPlanner(env, config)
  elif config['planner'] == 'tilt_house_building_1_deconstruct':
    return lambda env: TiltDeconstructPlanner(env, config)
  elif config['planner'] == 'tilt_house_building_3_deconstruct':
    return lambda env: TiltDeconstructPlanner(env, config)
  elif config['planner'] == 'tilt_house_building_4_deconstruct':
    return lambda env: TiltDeconstructPlanner(env, config)
  elif config['planner'] == 'tilt_improvise_house_building_2_deconstruct':
    return lambda env: TiltDeconstructPlanner(env, config)
  elif config['planner'] == 'tilt_improvise_house_building_3_deconstruct':
    return lambda env: TiltDeconstructPlanner(env, config)
  elif config['planner'] == 'tilt_improvise_house_building_5_deconstruct':
    return lambda env: TiltDeconstructPlanner(env, config)
  elif config['planner'] == 'tilt_improvise_house_building_6_deconstruct':
    return lambda env: TiltDeconstructPlanner(env, config)

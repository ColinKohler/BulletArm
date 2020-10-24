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
from helping_hands_rl_envs.planners.house_building_4_planner import HouseBuilding4Planner
from helping_hands_rl_envs.planners.improvise_house_building_2_planner import ImproviseHouseBuilding2Planner
from helping_hands_rl_envs.planners.improvise_house_building_3_planner import ImproviseHouseBuilding3Planner
from helping_hands_rl_envs.planners.deconstruct_planner import DeconstructPlanner

PLANNERS = {
  'random' : RandomPlanner,
  'multi_task' : MultiTaskPlanner,
  'play' : PlayPlanner,
  'block_picking' : BlockPickingPlanner,
  'block_stacking' : BlockStackingPlanner,
  'block_adjacent' : BlockAdjacentPlanner,
  'pyramid_stacking' : PyramidStackingPlanner,
  'brick_stacking' : BrickStackingPlanner,
  'house_building_1' : HouseBuilding1Planner,
  'house_building_2' : HouseBuilding2Planner,
  'house_building_3' : HouseBuilding3Planner,
  'house_building_4' : HouseBuilding4Planner,
  'improvise_house_building_2' : ImproviseHouseBuilding2Planner,
  'improvise_house_building_3' : ImproviseHouseBuilding3Planner,
  'house_building_1_deconstruct' : DeconstructPlanner,
  'house_building_2_deconstruct' : DeconstructPlanner,
  'house_building_3_deconstruct' : DeconstructPlanner,
  'house_building_4_deconstruct' : DeconstructPlanner,
  'house_building_x_deconstruct' : DeconstructPlanner,
  'improvise_house_building_2_deconstruct' : DeconstructPlanner,
  'improvise_house_building_3_deconstruct' : DeconstructPlanner,
  'improvise_house_building_4_deconstruct' : DeconstructPlanner,
}

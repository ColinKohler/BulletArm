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
from helping_hands_rl_envs.planners.ramp_block_stacking_planner import RampBlockStackingPlanner
from helping_hands_rl_envs.planners.ramp_deconstruct_planner import RampDeconstructPlanner
from helping_hands_rl_envs.planners.drawer_teapot_planner import DrawerTeapotPlanner
from helping_hands_rl_envs.planners.bowl_stacking_planner import BowlStackingPlanner
from helping_hands_rl_envs.planners.shelf_bowl_stacking_planner import ShelfBowlStackingPlanner
from helping_hands_rl_envs.planners.shelf_plate_stacking_planner import ShelfPlateStackingPlanner
from helping_hands_rl_envs.planners.drawer_shelf_plate_stacking_planner import DrawerShelfPlateStackingPlanner
from helping_hands_rl_envs.planners.block_bin_packing_planner import BlockBinPackingPlanner
from helping_hands_rl_envs.planners.bowl_spoon_cup_planner import BowlSpoonCupPlanner
from helping_hands_rl_envs.planners.bottle_tray_planner import BottleTrayPlanner
from helping_hands_rl_envs.planners.box_palletizing_planner import BoxPalletizingPlanner
from helping_hands_rl_envs.planners.bumpy_house_building_4_planner import BumpyHouseBuilding4Planner
from helping_hands_rl_envs.planners.covid_test_planner import CovidTestPlanner
from helping_hands_rl_envs.planners.close_loop_block_picking_planner import CloseLoopBlockPickingPlanner
from helping_hands_rl_envs.planners.close_loop_block_stacking_planner import CloseLoopBlockStackingPlanner
from helping_hands_rl_envs.planners.close_loop_block_pulling_planner import CloseLoopBlockPullingPlanner
from helping_hands_rl_envs.planners.close_loop_house_building_1_planner import CloseLoopHouseBuilding1Planner
from helping_hands_rl_envs.planners.close_loop_block_picking_corner_planner import CloseLoopBlockPickingCornerPlanner
from helping_hands_rl_envs.planners.close_loop_drawer_opening_planner import CloseLoopDrawerOpeningPlanner
from helping_hands_rl_envs.planners.close_loop_household_picking_cluttered_planner import CloseLoopHouseholdPickingClutteredPlanner
from helping_hands_rl_envs.planners.close_loop_household_pushing_planner import CloseLoopHouseholdPushingPlanner
from helping_hands_rl_envs.planners.close_loop_block_pushing_planner import CloseLoopBlockPushingPlanner

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
  'block_stacking_deconstruct': DeconstructPlanner,
  'house_building_1_deconstruct' : DeconstructPlanner,
  'house_building_2_deconstruct' : DeconstructPlanner,
  'house_building_3_deconstruct' : DeconstructPlanner,
  'house_building_4_deconstruct' : DeconstructPlanner,
  'house_building_x_deconstruct' : DeconstructPlanner,
  'improvise_house_building_2_deconstruct' : DeconstructPlanner,
  'improvise_house_building_3_deconstruct' : DeconstructPlanner,
  'improvise_house_building_discrete_deconstruct' : DeconstructPlanner,
  'improvise_house_building_random_deconstruct' : DeconstructPlanner,
  'ramp_block_stacking': RampBlockStackingPlanner,
  'ramp_block_stacking_deconstruct': RampDeconstructPlanner,
  'ramp_house_building_1_deconstruct': RampDeconstructPlanner,
  'ramp_house_building_2_deconstruct': RampDeconstructPlanner,
  'ramp_house_building_3_deconstruct': RampDeconstructPlanner,
  'ramp_house_building_4_deconstruct': RampDeconstructPlanner,
  'ramp_improvise_house_building_2_deconstruct': RampDeconstructPlanner,
  'ramp_improvise_house_building_3_deconstruct': RampDeconstructPlanner,
  'two_view_drawer_teapot': DrawerTeapotPlanner,
  'multi_view_drawer_teapot': DrawerTeapotPlanner,
  'cup_stacking': BlockStackingPlanner,
  'bowl_stacking': BowlStackingPlanner,
  'shelf_bowl_stacking': ShelfBowlStackingPlanner,
  'shelf_plate_stacking': ShelfPlateStackingPlanner,
  'drawer_shelf_plate_stacking': DrawerShelfPlateStackingPlanner,
  'block_bin_packing': BlockBinPackingPlanner,
  'random_block_picking': BlockPickingPlanner,
  'random_block_picking_clutter': BlockPickingPlanner,
  'random_household_picking': BlockPickingPlanner,
  'random_household_picking_clutter': BlockPickingPlanner,
  'bowl_spoon_cup': BowlSpoonCupPlanner,
  'bottle_tray': BottleTrayPlanner,
  'box_palletizing': BoxPalletizingPlanner,
  'bumpy_box_palletizing': BoxPalletizingPlanner,
  'bumpy_house_building_4': BumpyHouseBuilding4Planner,
  'covid_test':CovidTestPlanner,
  'close_loop_block_picking':CloseLoopBlockPickingPlanner,
  'close_loop_block_reaching':CloseLoopBlockPickingPlanner,
  'close_loop_block_stacking':CloseLoopBlockStackingPlanner,
  'close_loop_block_pulling':CloseLoopBlockPullingPlanner,
  'close_loop_house_building_1':CloseLoopHouseBuilding1Planner,
  'close_loop_block_picking_corner':CloseLoopBlockPickingCornerPlanner,
  'close_loop_drawer_opening':CloseLoopDrawerOpeningPlanner,
  'close_loop_household_picking':CloseLoopBlockPickingPlanner,
  'close_loop_clutter_picking':CloseLoopHouseholdPickingClutteredPlanner,
  'close_loop_household_pushing':CloseLoopHouseholdPushingPlanner,
  'close_loop_block_pushing':CloseLoopBlockPushingPlanner,
}

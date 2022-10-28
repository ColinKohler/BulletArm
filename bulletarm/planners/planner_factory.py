from bulletarm.planners import *

def getPlannerFn(env_type, planner_config):
  return lambda env: PLANNERS[env_type](env, planner_config)

PLANNERS = {
  'random' : RandomPlanner,
  'multi_task' : MultiTaskPlanner,
  'play' : PlayPlanner,
  'block_picking' : BlockPickingPlanner,
  'block_stacking' : BlockStackingPlanner,
  'pyramid_stacking' : PyramidStackingPlanner,
  'brick_stacking' : BrickStackingPlanner,
  'house_building_1' : HouseBuilding1Planner,
  'house_building_2' : HouseBuilding2Planner,
  'house_building_3' : HouseBuilding3Planner,
  'house_building_4' : HouseBuilding4Planner,
  'improvise_house_building_2' : ImproviseHouseBuilding2Planner,
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
  'cup_stacking': BlockStackingPlanner,
  'block_bin_packing': BlockBinPackingPlanner,
  'object_grasping': BlockPickingPlanner,
  'random_block_picking': BlockPickingPlanner,
  'random_block_picking_clutter': BlockPickingPlanner,
  'random_household_picking': BlockPickingPlanner,
  'clutter_picking': BlockPickingPlanner,
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
  'close_loop_block_pulling_corner':CloseLoopBlockPickingCornerPlanner,
  'close_loop_block_picking_corner':CloseLoopBlockPickingCornerPlanner,
  'close_loop_drawer_opening':CloseLoopDrawerOpeningPlanner,
  'close_loop_household_picking':CloseLoopBlockPickingPlanner,
  'close_loop_clutter_picking':CloseLoopHouseholdPickingClutteredPlanner,
  'close_loop_household_pushing':CloseLoopHouseholdPushingPlanner,
  'close_loop_block_pushing':CloseLoopBlockPushingPlanner,
  'close_loop_block_in_bowl':CloseLoopBlockInBowlPlanner,
  'close_loop_peg_insertion' : CloseLoopPegInsertionPlanner,
  'close_loop_mug_picking' : CloseLoopMugPickingPlanner,
  'force_block_picking' : CloseLoopBlockPickingPlanner,
  'force_block_reaching' : CloseLoopBlockPickingPlanner,
  'force_block_pulling' : CloseLoopBlockPullingPlanner,
  'force_block_pushing' : CloseLoopBlockPushingPlanner,
  'force_block_stacking' : CloseLoopBlockStackingPlanner,
  'force_block_pulling_corner' : CloseLoopBlockPickingCornerPlanner,
  'force_block_picking_corner' : CloseLoopBlockPickingCornerPlanner,
  'force_peg_insertion' : CloseLoopPegInsertionPlanner,
  'force_drawer_opening' : CloseLoopDrawerOpeningPlanner,
  'force_clutter_picking' : CloseLoopHouseholdPickingClutteredPlanner,
  'force_mug_picking' : CloseLoopMugPickingPlanner,
}

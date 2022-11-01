from bulletarm.envs.close_loop_envs import *
from bulletarm.envs.force_envs import *
from bulletarm.envs.block_structure_envs import *
from bulletarm.envs.deconstruct_envs import *
from bulletarm.envs.realistic_envs import *
from bulletarm.envs.ramp_envs import *
from bulletarm.envs.bumpy_envs import *

def createEnv(env):
  return lambda config: env(config)

CREATE_ENV_FNS = {
  'block_picking' : createEnv(BlockPickingEnv),
  'block_stacking' : createEnv(BlockStackingEnv),
  'brick_stacking' : createEnv(BrickStackingEnv),
  'pyramid_stacking' : createEnv(PyramidStackingEnv),
  'house_building_1' : createEnv(HouseBuilding1Env),
  'house_building_2' : createEnv(HouseBuilding2Env),
  'house_building_3' : createEnv(HouseBuilding3Env),
  'house_building_4' : createEnv(HouseBuilding4Env),
  'house_building_5' : createEnv(HouseBuilding5Env),
  'house_buliding_x' : createEnv(HouseBuildingXEnv),
  'improvise_house_building_2' : createEnv(ImproviseHouseBuilding2Env),
  'improvise_house_building_3' : createEnv(ImproviseHouseBuilding3Env),
  'improvise_house_building_discrete' : createEnv(ImproviseHouseBuildingDiscreteEnv),
  'improvise_house_building_random' : createEnv(ImproviseHouseBuildingRandomEnv),
  'block_stacking_deconstruct': createEnv(BlockStackingDeconstructEnv),
  'house_building_1_deconstruct' : createEnv(HouseBuilding1DeconstructEnv),
  'house_building_2_deconstruct' : createEnv(HouseBuilding2DeconstructEnv),
  'house_building_3_deconstruct' : createEnv(HouseBuilding3DeconstructEnv),
  'house_building_4_deconstruct' : createEnv(HouseBuilding4DeconstructEnv),
  'house_building_x_deconstruct' : createEnv(HouseBuildingXDeconstructEnv),
  'improvise_house_building_2_deconstruct' : createEnv(ImproviseHouseBuilding2DeconstructEnv),
  'improvise_house_building_3_deconstruct' : createEnv(ImproviseHouseBuilding3DeconstructEnv),
  'improvise_house_building_discrete_deconstruct' : createEnv(ImproviseHouseBuildingDiscreteDeconstructEnv),
  'improvise_house_building_random_deconstruct' : createEnv(ImproviseHouseBuildingRandomDeconstructEnv),
  'ramp_block_stacking': createEnv(RampBlockStackingEnv),
  'ramp_house_building_1': createEnv(RampHouseBuilding1Env),
  'ramp_house_building_2': createEnv(RampHouseBuilding2Env),
  'ramp_house_building_3': createEnv(RampHouseBuilding3Env),
  'ramp_house_building_4': createEnv(RampHouseBuilding4Env),
  'ramp_improvise_house_building_2': createEnv(RampImproviseHouseBuilding2Env),
  'ramp_improvise_house_building_3': createEnv(RampImproviseHouseBuilding3Env),
  'ramp_block_stacking_deconstruct': createEnv(RampBlockStackingDeconstructEnv),
  'ramp_house_building_1_deconstruct': createEnv(RampHouseBuilding1DeconstructEnv),
  'ramp_house_building_2_deconstruct': createEnv(RampHouseBuilding2DeconstructEnv),
  'ramp_house_building_3_deconstruct': createEnv(RampHouseBuilding3DeconstructEnv),
  'ramp_house_building_4_deconstruct': createEnv(RampHouseBuilding4DeconstructEnv),
  'ramp_improvise_house_building_2_deconstruct': createEnv(RampImproviseHouseBuilding2DeconstructEnv),
  'ramp_improvise_house_building_3_deconstruct': createEnv(RampImproviseHouseBuilding3DeconstructEnv),
  'object_grasping': createEnv(ObjectGraspingEnv),
  'block_bin_packing': createEnv(BlockBinPackingEnv),
  'random_block_picking': createEnv(RandomBlockPickingEnv),
  'random_household_picking': createEnv(RandomHouseholdPickingEnv),
  'random_block_picking_clutter': createEnv(RandomBlockPickingClutterEnv),
  'clutter_picking': createEnv(RandomHouseholdPickingClutterEnv),
  'bottle_tray': createEnv(BottleTrayEnv),
  'box_palletizing': createEnv(BoxPalletizingEnv),
  'bumpy_box_palletizing': createEnv(BumpyBoxPalletizingEnv),
  'bumpy_house_building_4': createEnv(BumpyHouseBuilding4Env),
  'covid_test': createEnv(CovidTestEnv),
  'close_loop_block_picking': createEnv(CloseLoopBlockPickingEnv),
  'close_loop_block_reaching': createEnv(CloseLoopBlockReachingEnv),
  'close_loop_block_stacking': createEnv(CloseLoopBlockStackingEnv),
  'close_loop_block_pulling': createEnv(CloseLoopBlockPullingEnv),
  'close_loop_house_building_1': createEnv(CloseLoopHouseBuilding1Env),
  'close_loop_block_pulling_corner': createEnv(CloseLoopBlockPullingCornerEnv),
  'close_loop_block_picking_corner': createEnv(CloseLoopBlockPickingCornerEnv),
  'close_loop_drawer_opening': createEnv(CloseLoopDrawerOpeningEnv),
  'close_loop_drawer_closing': createEnv(CloseLoopDrawerClosingEnv),
  'close_loop_household_picking': createEnv(CloseLoopHouseholdPickingEnv),
  'close_loop_clutter_picking': createEnv(CloseLoopHouseholdPickingClutteredEnv),
  'close_loop_block_pushing': createEnv(CloseLoopBlockPushingEnv),
  'close_loop_block_in_bowl': createEnv(CloseLoopBlockInBowlEnv),
  'close_loop_peg_insertion': createEnv(CloseLoopPegInsertionEnv),
  'close_loop_mug_picking': createEnv(CloseLoopMugPickingEnv),
  'force_block_picking': createEnv(ForceBlockPickingEnv),
  'force_block_reaching': createEnv(ForceBlockReachingEnv),
  'force_block_stacking': createEnv(ForceBlockStackingEnv),
  'force_block_pulling': createEnv(ForceBlockPullingEnv),
  'force_block_pushing': createEnv(ForceBlockPushingEnv),
  'force_block_picking_corner': createEnv(ForceBlockPickingCornerEnv),
  'force_block_pulling_corner': createEnv(ForceBlockPullingCornerEnv),
  'force_drawer_opening': createEnv(ForceDrawerOpeningEnv),
  'force_drawer_closing': createEnv(ForceDrawerClosingEnv),
  'force_clutter_picking': createEnv(ForceHouseholdPickingClutteredEnv),
  'force_block_in_bowl': createEnv(ForceBlockInBowlEnv),
  'force_peg_insertion': createEnv(ForcePegInsertionEnv),
  'force_mug_picking': createEnv(ForceMugPickingEnv),
}

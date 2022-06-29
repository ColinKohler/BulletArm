from bulletarm.envs.multi_task_env import createMultiTaskEnv

from bulletarm.envs.deconstruct_envs.block_stacking_deconstruct_env import createBlockStackingDeconstructEnv
from bulletarm.envs.deconstruct_envs.house_building_1_deconstruct_env import createHouseBuilding1DeconstructEnv
from bulletarm.envs.deconstruct_envs.house_building_2_deconstruct_env import createHouseBuilding2DeconstructEnv
from bulletarm.envs.deconstruct_envs.house_building_3_deconstruct_env import createHouseBuilding3DeconstructEnv
from bulletarm.envs.deconstruct_envs.house_building_4_deconstruct_env import createHouseBuilding4DeconstructEnv
from bulletarm.envs.deconstruct_envs.house_building_x_deconstruct_env import createHouseBuildingXDeconstructEnv
from bulletarm.envs.deconstruct_envs.improvise_house_building_2_deconstruct_env import createImproviseHouseBuilding2DeconstructEnv
from bulletarm.envs.deconstruct_envs.improvise_house_building_3_deconstruct_env import createImproviseHouseBuilding3DeconstructEnv
from bulletarm.envs.deconstruct_envs.improvise_house_building_discrete_deconstruct_env import createImproviseHouseBuildingDiscreteDeconstructEnv
from bulletarm.envs.deconstruct_envs.improvise_house_building_random_deconstruct_env import createImproviseHouseBuildingRandomDeconstructEnv

from bulletarm.envs.realistic_envs.object_grasping import createObjectGrasping
from bulletarm.envs.realistic_envs.block_picking_env import createBlockPickingEnv
from bulletarm.envs.realistic_envs.block_bin_packing_env import createBlockBinPackingEnv
from bulletarm.envs.realistic_envs.random_block_picking_env import createRandomBlockPickingEnv
from bulletarm.envs.realistic_envs.random_household_picking_env import createRandomHouseholdPickingEnv
from bulletarm.envs.realistic_envs.random_block_picking_clutter_env import createRandomBlockPickingClutterEnv
from bulletarm.envs.realistic_envs.random_household_picking_clutter_env import createRandomHouseholdPickingClutterEnv
from bulletarm.envs.realistic_envs.bottle_tray_env import createBottleTrayEnv
from bulletarm.envs.realistic_envs.box_palletizing_env import createBoxPalletizingEnv
from bulletarm.envs.realistic_envs.covid_test_env import createCovidTestEnv

from bulletarm.envs.ramp_envs.ramp_block_stacking_env import createRampBlockStackingEnv
from bulletarm.envs.ramp_envs.ramp_house_building_1_env import createRampHouseBuilding1Env
from bulletarm.envs.ramp_envs.ramp_house_building_2_env import createRampHouseBuilding2Env
from bulletarm.envs.ramp_envs.ramp_house_building_3_env import createRampHouseBuilding3Env
from bulletarm.envs.ramp_envs.ramp_house_building_4_env import createRampHouseBuilding4Env
from bulletarm.envs.ramp_envs.ramp_improvise_house_building_2_env import createRampImproviseHouseBuilding2Env
from bulletarm.envs.ramp_envs.ramp_improvise_house_building_3_env import createRampImproviseHouseBuilding3Env
from bulletarm.envs.ramp_envs.ramp_block_stacking_deconstruct_env import createRampBlockStackingDeconstructEnv
from bulletarm.envs.ramp_envs.ramp_house_building_1_deconstruct_env import createRampHouseBuilding1DeconstructEnv
from bulletarm.envs.ramp_envs.ramp_house_building_2_deconstruct_env import createRampHouseBuilding2DeconstructEnv
from bulletarm.envs.ramp_envs.ramp_house_building_3_deconstruct_env import createRampHouseBuilding3DeconstructEnv
from bulletarm.envs.ramp_envs.ramp_house_building_4_deconstruct_env import createRampHouseBuilding4DeconstructEnv
from bulletarm.envs.ramp_envs.ramp_improvise_house_building_2_deconstruct_env import createRampImproviseHouseBuilding2DeconstructEnv
from bulletarm.envs.ramp_envs.ramp_improvise_house_building_3_deconstruct_env import createRampImproviseHouseBuilding3DeconstructEnv

from bulletarm.envs.bumpy_envs.bumpy_box_palletizing_env import createBumpyBoxPalletizingEnv
from bulletarm.envs.bumpy_envs.bumpy_house_building_4_env import createBumpyHouseBuilding4Env

from bulltetarm.envs.close_loop_envs import *
from bulletarm.envs.block_structure_envs import *

def createEnv(env):
  return lambda config: env(config)

CREATE_ENV_FNS = {
  'block_picking' : createBlockPickingEnv,
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
  'block_stacking_deconstruct': createBlockStackingDeconstructEnv,
  'house_building_1_deconstruct' : createHouseBuilding1DeconstructEnv,
  'house_building_2_deconstruct' : createHouseBuilding2DeconstructEnv,
  'house_building_3_deconstruct' : createHouseBuilding3DeconstructEnv,
  'house_building_4_deconstruct' : createHouseBuilding4DeconstructEnv,
  'house_building_x_deconstruct' : createHouseBuildingXDeconstructEnv,
  'improvise_house_building_2_deconstruct' : createImproviseHouseBuilding2DeconstructEnv,
  'improvise_house_building_3_deconstruct' : createImproviseHouseBuilding3DeconstructEnv,
  'improvise_house_building_discrete_deconstruct' : createImproviseHouseBuildingDiscreteDeconstructEnv,
  'improvise_house_building_random_deconstruct' : createImproviseHouseBuildingRandomDeconstructEnv,
  'multi_task' : createMultiTaskEnv,
  'ramp_block_stacking': createRampBlockStackingEnv,
  'ramp_house_building_1': createRampHouseBuilding1Env,
  'ramp_house_building_2': createRampHouseBuilding2Env,
  'ramp_house_building_3': createRampHouseBuilding3Env,
  'ramp_house_building_4': createRampHouseBuilding4Env,
  'ramp_improvise_house_building_2': createRampImproviseHouseBuilding2Env,
  'ramp_improvise_house_building_3': createRampImproviseHouseBuilding3Env,
  'ramp_block_stacking_deconstruct': createRampBlockStackingDeconstructEnv,
  'ramp_house_building_1_deconstruct': createRampHouseBuilding1DeconstructEnv,
  'ramp_house_building_2_deconstruct': createRampHouseBuilding2DeconstructEnv,
  'ramp_house_building_3_deconstruct': createRampHouseBuilding3DeconstructEnv,
  'ramp_house_building_4_deconstruct': createRampHouseBuilding4DeconstructEnv,
  'ramp_improvise_house_building_2_deconstruct': createRampImproviseHouseBuilding2DeconstructEnv,
  'ramp_improvise_house_building_3_deconstruct': createRampImproviseHouseBuilding3DeconstructEnv,
  'object_grasping': createObjectGrasping,
  'block_bin_packing': createBlockBinPackingEnv,
  'random_block_picking': createRandomBlockPickingEnv,
  'random_household_picking': createRandomHouseholdPickingEnv,
  'random_block_picking_clutter': createRandomBlockPickingClutterEnv,
  'random_household_picking_clutter': createRandomHouseholdPickingClutterEnv,
  'bottle_tray': createBottleTrayEnv,
  'box_palletizing': createBoxPalletizingEnv,
  'bumpy_box_palletizing': createBumpyBoxPalletizingEnv,
  'bumpy_house_building_4': createBumpyHouseBuilding4Env,
  'covid_test': createCovidTestEnv,
  'close_loop_block_picking': createEnv(CloseLoopBlockPickingEnv),
  'close_loop_block_reaching': createEnv(CloseLoopBlockReachingEnv),
  'close_loop_block_stacking': createEnv(CloseLoopBlockStackingEnv),
  'close_loop_block_pulling': createEnv(CloseLoopBlockPullingEnv),
  'close_loop_house_building_1': createEnv(CloseLoopHouseBuilding1Env),
  'close_loop_block_picking_corner': createEnv(CloseLoopBlockPickingCornerEnv),
  'close_loop_drawer_opening': createEnv(CloseLoopDrawerOpeningEnv),
  'close_loop_household_picking': createEnv(CloseLoopHouseholdPickingEnv),
  'close_loop_clutter_picking': createEnv(CloseLoopHouseholdPickingClutteredEnv),
  'close_loop_block_pushing': createEnv(CloseLoopBlockPushingEnv),
  'close_loop_block_in_bowl': createEnv(CloseLoopBlockInBowlEnv),
}

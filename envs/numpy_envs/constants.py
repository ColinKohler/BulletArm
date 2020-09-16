from helping_hands_rl_envs.envs.numpy_envs.block_picking_env import createBlockPickingEnv
from helping_hands_rl_envs.envs.numpy_envs.block_stacking_env import createBlockStackingEnv
from helping_hands_rl_envs.envs.numpy_envs.block_adjacent_env import createBlockAdjacentEnv
from helping_hands_rl_envs.envs.numpy_envs.brick_stacking_env import createBrickStackingEnv
from helping_hands_rl_envs.envs.numpy_envs.pyramid_stacking_env import createPyramidStackingEnv
from helping_hands_rl_envs.envs.numpy_envs.house_building_1_env import createHouseBuilding1Env
from helping_hands_rl_envs.envs.numpy_envs.house_building_2_env import createHouseBuilding2Env
from helping_hands_rl_envs.envs.numpy_envs.house_building_3_env import createHouseBuilding3Env
from helping_hands_rl_envs.envs.numpy_envs.house_building_4_env import createHouseBuilding4Env
from helping_hands_rl_envs.envs.numpy_envs.house_building_5_env import createHouseBuilding5Env

CREATE_NUMPY_ENV_FNS = {
  'block_picking' : createBlockPickingEnv,
  'block_stacking' : createBlockStackingEnv,
  'block_adjacent' : createBlockAdjacentEnv,
  'brick_stacking' : createBrickStackingEnv,
  'pyramid_stacking' : createPyramidStackingEnv,
  'house_building_1' : createHouseBuilding1Env,
  'house_building_2' : createHouseBuilding2Env,
  'house_building_3' : createHouseBuilding3Env,
  'house_building_4' : createHouseBuilding4Env,
  'house_building_5' : createHouseBuilding5Env,
}

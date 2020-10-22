import numpy as np

from helping_hands_rl_envs.envs.pybullet_envs.block_picking_env import createBlockPickingEnv
from helping_hands_rl_envs.envs.pybullet_envs.block_stacking_env import createBlockStackingEnv
from helping_hands_rl_envs.envs.pybullet_envs.block_adjacent_env import createBlockAdjacentEnv
from helping_hands_rl_envs.envs.pybullet_envs.brick_stacking_env import createBrickStackingEnv
from helping_hands_rl_envs.envs.pybullet_envs.pyramid_stacking_env import createPyramidStackingEnv
from helping_hands_rl_envs.envs.pybullet_envs.house_building_1_env import createHouseBuilding1Env
from helping_hands_rl_envs.envs.pybullet_envs.house_building_2_env import createHouseBuilding2Env
from helping_hands_rl_envs.envs.pybullet_envs.house_building_3_env import createHouseBuilding3Env
from helping_hands_rl_envs.envs.pybullet_envs.house_building_4_env import createHouseBuilding4Env
from helping_hands_rl_envs.envs.pybullet_envs.house_building_5_env import createHouseBuilding5Env
from helping_hands_rl_envs.envs.pybullet_envs.house_building_x_env import createHouseBuildingXEnv
from helping_hands_rl_envs.envs.pybullet_envs.improvise_house_building_2_env import createImproviseHouseBuilding2Env
from helping_hands_rl_envs.envs.pybullet_envs.improvise_house_building_3_env import createImproviseHouseBuilding3Env
from helping_hands_rl_envs.envs.pybullet_envs.improvise_house_building_4_env import createImproviseHouseBuilding4Env
from helping_hands_rl_envs.envs.pybullet_envs.house_building_1_deconstruct_env import createHouseBuilding1DeconstructEnv
from helping_hands_rl_envs.envs.pybullet_envs.house_building_4_deconstruct_env import createHouseBuilding4DeconstructEnv
from helping_hands_rl_envs.envs.pybullet_envs.house_building_x_deconstruct_env import createHouseBuildingXDeconstructEnv
from helping_hands_rl_envs.envs.pybullet_envs.improvise_house_building_3_deconstruct_env import createImproviseHouseBuilding3DeconstructEnv
from helping_hands_rl_envs.envs.pybullet_envs.improvise_house_building_4_deconstruct_env import createImproviseHouseBuilding4DeconstructEnv
from helping_hands_rl_envs.envs.pybullet_envs.random_picking_env import createRandomPickingEnv
from helping_hands_rl_envs.envs.pybullet_envs.random_stacking_env import createRandomStackingEnv
from helping_hands_rl_envs.envs.pybullet_envs.multi_task_env import createMultiTaskEnv

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

CREATE_PYBULLET_ENV_FNS = {
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
  'house_buliding_x' : createHouseBuildingXEnv,
  'improvise_house_building_2' : createImproviseHouseBuilding2Env,
  'improvise_house_building_3' : createImproviseHouseBuilding3Env,
  'improvise_house_building_4' : createImproviseHouseBuilding4Env,
  'house_building_1_deconstruct' : createHouseBuilding1DeconstructEnv,
  'house_building_4_deconstruct' : createHouseBuilding4DeconstructEnv,
  'house_building_x_deconstruct' : createHouseBuildingXDeconstructEnv,
  'improvise_house_building_3_deconstruct' : createImproviseHouseBuilding3DeconstructEnv,
  'improvise_house_building_4_deconstruct' : createImproviseHouseBuilding4DeconstructEnv,
  'random_picking' : createRandomPickingEnv,
  'random_stacking' : createRandomStackingEnv,
  'multi_task' : createMultiTaskEnv,
}

DEFAULT_CONFIG = {
  'robot' : 'ur5',
  'pos_candidate' : None,
  'perfect_grasp' : False,
  'perfect_place' : False,
  'workspace_check' : 'box',
  'in_hand_size' : 24,
  'in_hand_mode' : 'sub',
  'num_random_objects' : 0,
  'check_random_obj_valid' : False,
  'action_sequence' : 'pxyr',
  'simulate_grasp' : True,
  'workspace' : np.array([[0.30, 0.60], [-0.15, 0.15], [0, 1]]),
  'object_scale_range': (0.60, 0.70),
  'max_steps' : 10,
  'obs_size' : 128,
  'fast_mode' : True,
  'render' : False,
  'num_solver_iterations' : 200,
  'solver_residual_threshold' : 1e-7,
  'random_orientation': False,
  'num_objects': 1,
  'reward_type': 'sparse',
  'object_type': 'cube',
}

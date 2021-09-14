import os
import numpy as np

ROOT = os.path.abspath(os.getcwd())
RESULTS_PATH = ROOT + '/data/results'
BENCHMARK_RESULT_PATH = ROOT + '/data/benchmark_results'

ONE_GPU_CONFIG = {
  'num_sampler_workers' : 4,
}
TWO_GPU_CONFIG = {
  'num_sampler_workers' : 8,
}
THREE_GPU_CONFIG = {
  'num_sampler_workers' : 12,
}
FOUR_GPU_CONFIG = {
  'num_sampler_workers' : 12,
}
GPU_CONFIG = {
  1 : ONE_GPU_CONFIG,
  2 : TWO_GPU_CONFIG,
  3 : THREE_GPU_CONFIG,
  4 : FOUR_GPU_CONFIG
}

OBS_SIZE = 128
DEICTIC_OBS_SIZE = 64
HAND_OBS_SIZE = 24

ROBOT = 'kuka'
WORKSPACE = np.array([[0.2, 0.6], [-0.20, 0.20], [0, 1]])
WORKSPACE_SIZE = WORKSPACE[0,1] - WORKSPACE[0,0]
OBS_RESOLUTION = WORKSPACE_SIZE / OBS_SIZE

BLOCK_STACKING_3_ENV_CONFIG = {
  'workspace': WORKSPACE,
  'max_steps': 10,
  'obs_size': OBS_SIZE,
  'in_hand_size': HAND_OBS_SIZE,
  'action_sequence': 'pxyr',
  'fast_mode': True,
  'simulate_grasps': True,
  'physics_mode' : 'slow',
  'robot': ROBOT,
  'num_objects' : 3,
  'object_scale_range': (0.8, 0.8),
}

HOUSE_BUILDING_2_ENV_CONFIG = {
  'workspace': WORKSPACE,
  'max_steps': 10,
  'obs_size': OBS_SIZE,
  'in_hand_size': HAND_OBS_SIZE,
  'action_sequence': 'pxyr',
  'fast_mode': True,
  'simulate_grasps': True,
  'physics_mode' : 'slow',
  'robot': ROBOT,
  'num_objects': 3,
  'object_scale_range': (0.8, 0.8),
}

BOTTLE_TRAY_ENV_CONFIG = {
  'workspace': WORKSPACE,
  'max_steps': 10,
  'obs_size': OBS_SIZE,
  'in_hand_size': HAND_OBS_SIZE,
  'action_sequence': 'pxyr',
  'fast_mode': True,
  'simulate_grasps': True,
  'physics_mode' : 'slow',
  'robot': ROBOT,
  'num_objects': 6,
  'object_init_space_check' : 'point',
  'object_scale_range': (0.8, 0.8),
  'kuka_adjust_gripper_offset' : 0.0025
}

BIN_PACKING_ENV_CONFIG = {
  'workspace': WORKSPACE,
  'max_steps': 10,
  'obs_size': OBS_SIZE,
  'in_hand_size': HAND_OBS_SIZE,
  'action_sequence': 'pxyr',
  'fast_mode': True,
  'simulate_grasps': True,
  'physics_mode' : 'slow',
  'robot': ROBOT,
  'num_objects': 6,
  'object_scale_range': (0.75, 0.75),
}

ENV_CONFIGS = {
  'block_stacking_3' : BLOCK_STACKING_3_ENV_CONFIG,
  'block_stacking_3_deconstruct' : BLOCK_STACKING_3_ENV_CONFIG,
  'house_building_2' : HOUSE_BUILDING_2_ENV_CONFIG,
  'house_building_2_deconstruct' : HOUSE_BUILDING_2_ENV_CONFIG,
  'bottle_tray' : BOTTLE_TRAY_ENV_CONFIG,
  'bin_packing' : BIN_PACKING_ENV_CONFIG,
}

ENV_TYPES = {
  'block_stacking_3' : 'block_stacking',
  'block_stacking_3_deconstruct' : 'block_stacking_deconstruct',
  'house_building_2' : 'house_building_2',
  'house_building_2_deconstruct' : 'house_building_2_deconstruct',
  'bottle_tray' : 'bottle_tray',
  'bin_packing' : 'block_bin_packing',
}

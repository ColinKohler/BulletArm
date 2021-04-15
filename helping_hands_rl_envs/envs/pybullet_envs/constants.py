import numpy as np

DEFAULT_CONFIG = {
  'robot' : 'ur5',
  'pos_candidate' : None,
  'perfect_grasp' : False,
  'perfect_place' : False,
  'workspace_check' : 'box',
  'in_hand_size' : 24,
  'in_hand_mode' : 'sub',
  'num_random_objects' : 0,
  'random_orientation' : False,
  'check_random_obj_valid' : False,
  'action_sequence' : 'pxyr',
  'simulate_grasp' : True,
  'workspace' : np.array([[0.30, 0.60], [-0.15, 0.15], [0, 1]]),
  'object_scale_range': (0.60, 0.70),
  'max_steps' : 10,
  'obs_size' : 128,
  'fast_mode' : True,
  'render' : False,
  'physics_mode' : 'fast',
  'reward_type' : 'sparse',
  'num_objects' : 1,
  'object_type' : 'cube',
  'hard_reset_freq': 1,
  'view_type': 'render', # render or camera
  'min_object_distance': None,
  'min_boarder_padding': None,
  # The random offset range for each object when generating the goal structure. This will help to reduce the domain gap
  # (because when constructing, the objects are aligned less perfectly), but will also decrease the optimality of the expert.
  # This is the sum of the + and - amount, e.g., for 0.005, the offset will be randomly sampled from -0.0025 to 0.0025
  'deconstruct_init_offset': 0,
  # If approaching the object top-down in picking. If False, the arm will approach along the direction of the ee
  'pick_top_down_approach': False,
  # If approaching the object top-down in placing. If False, the arm will approach along the direction of the ee
  'place_top_down_approach': False,
  # If True, constraint the top-down rotation in a 180 degree range
  'half_rotation': True,
  # If true, adjusting the gripper command w.r.t. the object grasped after moving to pre pose, otherwise adjusting the
  # gripper command before moving to pre pose. Adjusting after lifting will create more chance for a grasp, but while
  # moving to pre pose the gripper will shift around. Adjusting before lifting will make the gripper more stable while
  # moving to the pre pose, but will reduce the chance for a grasp, especially in the cluttered scene.
  'adjust_gripper_after_lift': False
}

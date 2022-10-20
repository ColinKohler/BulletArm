import numpy as np

DEFAULT_CONFIG = {
  # The type of robot to use in the simulator. Currently supports: kuka, panda, ur5, ur5_robotiq
  'robot' : 'kuka',
  # Check object out of bound using the COM or the bounding box. Choices: 'point', 'box'
  'workspace_check' : 'point',
  # The pixel size of the in-hand image
  'in_hand_size' : 24,
  # The type of the in-hand image:
  # 'raw': will use the raw crop
  # 'sub': will adjust the in-hand image to remove background objects by subtracting the max of the new observation
  #        after picking
  # 'proj': will generate the 3-channel orthographic projection image
  'in_hand_mode' : 'raw',
  # Whether or not to initialize the objects with random orientation
  'random_orientation' : True,
  # The action space. Consists of the following substrings:
  # p: the gripper motion;
  # x, y, z: the position of the gripper;
  # r: the rotation of the gripper;
  # rrr: the 3-dimensional rotation of the gripper
  # example: pxyr, pxyzrrr
  'action_sequence' : 'pxyr',
  # The workspace
  'workspace' : np.array([[0.25, 0.65], [-0.2, 0.2], [0, 1]]),
  # The scale of the objects
  'object_scale_range': (0.60, 0.70),
  # The maximal number of steps per episode
  'max_steps' : 10,
  # The observation size in pixel
  'obs_size' : 128,
  # If True, disable the physics simulation when the arm is moving from pre pose to home pose to speed up the simulation
  'fast_mode' : True,
  # If True, render the GUI of pybullet
  'render' : False,
  # The physics parameters
  'physics_mode' : 'fast',
  # The reward type
  'reward_type' : 'sparse',
  # The number of objects in the environment
  'num_objects' : 1,
  # The type of object in the environment. Currently only valid for block stacking. Choices: cube, cylinder
  'object_type' : 'cube',
  # The number of episodes to run a hard reset of pybullet
  'hard_reset_freq': 1000,
  # The type of the observation. Choices: 'render_center', 'render_center_height', 'render_fix', 'camera_center_xyzr',
  #                                       'camera_center_xyr', 'camera_center_xyz', 'camera_center_xy', 'camera_fix',
  #                                       'camera_center_xyr_height', 'camera_center_xyz_height',
  #                                       'camera_center_xy_height', 'camera_fix_height', 'camera_center_z',
  #                                       'camera_center_z_height', 'pers_center_xyz'
  'view_type': 'camera_center_xyz',
  # The minimal distance between objects in initialization
  'min_object_distance': None,
  # The minimal distance to the workspace boarder in initialization
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
  'adjust_gripper_after_lift': False,
  # The offset when adjusting gripper commands after gripper closes at an object. A bigger value increases the chance
  # for a grasp, but reduces the stability while holding it. Recommended value 0.01 or 0.001
  'kuka_adjust_gripper_offset': 0.01,

  ## close-loop env parameters ##
  # whether to include a tray in close loop env
  'close_loop_tray': False,
  # The ratio between the length covered in the observation to the size of the workspace
  'view_scale': 1.5,
  # The type of observation. Choices: 'pixel', 'vec'
  'obs_type': 'pixel',
  # workspace configurations to add a black area covering the workspace, removing the blue/white grid on the ground,
  # and so on. Takes in a list of keywords. Valid keywords:
  #   'white_plane': use a white plane instead of the default white/blue grid plane from pybullet.
  #   'black_workspace': add a black visual shape covering the workspace.
  #   'trans_plane': make the plane transparent.
  #   'trans_robot': make the robot transparent.
  # Example: 'workspace_option': ['white_plane', 'black_workspace']
  'workspace_option': [],

  ## Deprecated parameters ##
  'pos_candidate': None,
  'perfect_grasp': False,
  'perfect_place': False,
  'num_random_objects': 0,
  'check_random_obj_valid': False,
  'simulate_grasp': True,
}

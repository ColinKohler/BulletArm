import os
from collections import namedtuple
from attrdict import AttrDict

import bulletarm

class NoValidPositionException(Exception):
  pass

# File paths
URDF_PATH = os.path.join(os.path.dirname(bulletarm.__file__), 'pybullet/urdf/')
OBJECTS_PATH = os.path.join(URDF_PATH, 'object')
KUKA_PATH = os.path.join(URDF_PATH, 'kuka/kuka_with_gripper2.sdf')
PANDA_PATH = os.path.join(URDF_PATH, 'franka_panda/panda.urdf')
UR5_ROBOTIQ_PATH = os.path.join(URDF_PATH, 'ur5/ur5_robotiq_85_gripper_fake.urdf')
UR5_HYDROSTATIC_PATH = os.path.join(URDF_PATH, 'ur5/ur5_hydrostatic.urdf')
UR5_OPENHAND_VF_PATH = os.path.join(URDF_PATH, 'ur5/ur5_openhand_vf.urdf')

# Shape types
CUBE = 0
SPHERE = 1
CYLINDER = 2
CONE = 3
BRICK = 4
TRIANGLE = 5
ROOF = 6
RANDOM = 7
TEAPOT = 8
TEAPOT_LID = 9
CUP = 10
BOWL = 11
PLATE = 12
RANDOM_BLOCK = 13
RANDOM_HOUSEHOLD = 14
SPOON = 15
BOTTLE = 16
BOX = 17
PALLET = 18
TEST_TUBE = 19
SWAB = 20
FLAT_BLOCK = 21
RANDOM_HOUSEHOLD200 = 22
GRASP_NET_OBJ = 23
SQUARE_PEG = 24
RANDOM = -1

# Motion primatives
NUM_PRIMATIVES = 2
PICK_PRIMATIVE = 0
PLACE_PRIMATIVE = 1
PULL_PRIMATIVE = 2
PUSH_PRIMATIVE = 3

# Robots
JointInfo = namedtuple("JointInfo",
                       ["id", "name", "type", "lower_limit", "upper_limit", "max_force", "max_velocity"])
JOINT_TYPES = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]

z_scale_1 = 1
z_scale_2 = 2

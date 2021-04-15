class NoValidPositionException(Exception):
  pass

# File paths
URDF_PATH = 'simulators/urdf/object/'

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

# Motion primatives
NUM_PRIMATIVES = 2
PICK_PRIMATIVE = 0
PLACE_PRIMATIVE = 1
PULL_PRIMATIVE = 2
PUSH_PRIMATIVE = 3

z_scale_1 = 1
z_scale_2 = 2
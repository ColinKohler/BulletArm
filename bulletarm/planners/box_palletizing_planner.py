import numpy as np
from bulletarm.planners.block_structure_base_planner import BlockStructureBasePlanner
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils import transformations

class BoxPalletizingPlanner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(BoxPalletizingPlanner, self).__init__(env, config)

  def pickLargestObjOnTop(self, objects=None):
    if objects is None: objects = self.env.objects
    objects, object_poses = self.getSizeSortedObjPoses(objects=objects)

    x, y, z, rx, ry, rz = object_poses[0][0], object_poses[0][1], object_poses[0][2], object_poses[0][3], object_poses[0][4], object_poses[0][5]
    for obj, pose in zip(objects, object_poses):
      if self.isObjOnTop(obj):
        x, y, z, rx, ry, rz = pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]
        break

    T = transformations.euler_matrix(rx, ry, rz)
    x, y, z = np.array([x, y, z]) + T[:3, 2] * self.env.block_scale_range[1] * 0.01

    return self.encodeAction(constants.PICK_PRIMATIVE, x, y, z, (rz, ry, rx))

  def getPickingAction(self):
    obj_on_ground = list(filter(lambda o: self.env._isObjOnGround(o), self.env.objects))
    if len(obj_on_ground) == 0:
      obj_on_ground = [self.env.objects[-1]]
    return self.pickLargestObjOnTop(obj_on_ground)

  def getPlacingAction(self):
    n_level1, n_level2, n_level3 = self.env.getNEachLevel()
    if n_level1 < 6:
      x, y = self.env.odd_place_pos_candidate[n_level1]
      z = self.env.place_offset + self.env.box_height * 0.8
      r = self.env.pallet_rz + np.pi / 2
    elif n_level2 < 6:
      x, y = self.env.even_place_pos_candidate[n_level2]
      z = self.env.place_offset + self.env.box_height * 1.8
      r = self.env.pallet_rz
    else:
      x, y = self.env.odd_place_pos_candidate[n_level3]
      z = self.env.place_offset + self.env.box_height * 2.8
      r = self.env.pallet_rz + np.pi / 2
    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)


  def getStepsLeft(self):
    n_level1, n_level2, n_level3 = self.env.getNEachLevel()
    return 2*(self.env.num_obj - n_level1 - n_level2 - n_level3) - int(self.isHolding())

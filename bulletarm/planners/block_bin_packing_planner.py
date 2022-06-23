import numpy as np
from scipy import ndimage
import numpy.random as npr
import skimage.transform as sk_transform

from bulletarm.planners.base_planner import BasePlanner
from bulletarm.planners.block_structure_base_planner import BlockStructureBasePlanner
from bulletarm.pybullet.utils import constants

class BlockBinPackingPlanner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(BlockBinPackingPlanner, self).__init__(env, config)

  def getPickingAction(self):
    # return self.pickSecondTallestObjOnTop(self.env.getObjsOutsideBox())
    return self.pickLargestObjOnTop(self.env.getObjsOutsideBox())

  def getPlacingAction(self):
    workspace_center = self.env.workspace.mean(1)[:2]
    box_corner_pos = np.array([[-self.env.box_size[0]/2, -self.env.box_size[1]/2],
                               [self.env.box_size[0]/2, self.env.box_size[1]/2]])
    R = np.array([[np.cos(-self.env.box_rz), -np.sin(-self.env.box_rz)],
                  [np.sin(-self.env.box_rz), np.cos(-self.env.box_rz)]])
    transformed_box_pos = R.dot(np.array([self.env.box_pos[:2] - workspace_center]).T).T + workspace_center
    transformed_corner_pos = box_corner_pos + transformed_box_pos
    transformed_heightmap = sk_transform.rotate(self.env.heightmap, np.rad2deg(-self.env.box_rz))
    box_pixel_min = self.env._getPixelsFromPos(transformed_corner_pos[0, 0]+0.05, transformed_corner_pos[0, 1]+0.03)
    box_pixel_max = self.env._getPixelsFromPos(transformed_corner_pos[1, 0]-0.05, transformed_corner_pos[1, 1]-0.03)
    box_pixel_min = list(map(lambda x: int(x), box_pixel_min))
    box_pixel_max = list(map(lambda x: int(x), box_pixel_max))
    heightmap_box = transformed_heightmap[box_pixel_min[0]:box_pixel_max[0], box_pixel_min[1]:box_pixel_max[1]]
    # avg_heightmap = ndimage.uniform_filter(transformed_heightmap, 0.1*self.env.block_scale_range[1]//self.env.heightmap_resolution, mode='nearest')
    avg_heightmap_box = ndimage.uniform_filter(heightmap_box, 0.1*self.env.block_scale_range[1]//self.env.heightmap_resolution, mode='nearest')
    min_pixel = np.argmin(avg_heightmap_box)
    min_pixel = min_pixel // avg_heightmap_box.shape[1], min_pixel % avg_heightmap_box.shape[1]
    min_pixel = np.array(min_pixel) + box_pixel_min
    x, y = self.env._getPosFromPixels(*min_pixel)
    z = self.env.place_offset
    r = self.env.box_rz + np.pi/2
    R = np.array([[np.cos(self.env.box_rz), -np.sin(self.env.box_rz)],
                  [np.sin(self.env.box_rz), np.cos(self.env.box_rz)]])
    back_transform_xy = R.dot(np.array([np.array([x, y]) - workspace_center]).T).T + workspace_center
    x, y = back_transform_xy[0]

    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)


  def getStepsLeft(self):
    return 100

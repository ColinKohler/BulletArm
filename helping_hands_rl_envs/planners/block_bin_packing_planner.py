import numpy as np
from scipy import ndimage
import numpy.random as npr
import matplotlib.pyplot as plt

from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.planners.block_structure_base_planner import BlockStructureBasePlanner
from helping_hands_rl_envs.simulators import constants

import matplotlib.pyplot as plt

class BlockBinPackingPlanner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(BlockBinPackingPlanner, self).__init__(env, config)

  def getPickingAction(self):
    # return self.pickSecondTallestObjOnTop(self.env.getObjsOutsideBox())
    return self.pickLargestObjOnTop(self.env.getObjsOutsideBox())

  def getPlacingAction(self):
    box_pixel_min = self.env._getPixelsFromPos(self.env.box_range[0, 0]+0.03, self.env.box_range[1, 0]+0.03)
    box_pixel_max = self.env._getPixelsFromPos(self.env.box_range[0, 1]-0.03, self.env.box_range[1, 1]-0.03)
    box_pixel_min = list(map(lambda x: int(x), box_pixel_min))
    box_pixel_max = list(map(lambda x: int(x), box_pixel_max))
    avg_heightmap = ndimage.uniform_filter(self.env.heightmap, 0.1//self.env.heightmap_resolution, mode='nearest')
    avg_heightmap_box = avg_heightmap[box_pixel_min[0]:box_pixel_max[0], box_pixel_min[1]:box_pixel_max[1]]
    min_pixel = np.argmin(avg_heightmap_box)
    min_pixel = min_pixel // avg_heightmap_box.shape[1], min_pixel % avg_heightmap_box.shape[1]

    # plt.imshow(avg_heightmap_box)
    # plt.scatter(min_pixel[1], min_pixel[0], c='r')
    # plt.show()

    min_pixel = np.array(min_pixel) + box_pixel_min
    x, y = self.env._getPosFromPixels(*min_pixel)
    z = self.env.place_offset
    # r = np.pi*np.random.random_sample() if self.random_orientation else 0
    r = np.pi/2

    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)


  def getStepsLeft(self):
    return 100
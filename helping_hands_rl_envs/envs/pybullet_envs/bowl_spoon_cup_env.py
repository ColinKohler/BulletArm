import numpy as np
import pybullet as pb
from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.constants import NoValidPositionException
from helping_hands_rl_envs.simulators.pybullet.equipments.blanket import Blanket

class BowlSpoonCupEnv(PyBulletEnv):
  def __init__(self, config):
    super(BowlSpoonCupEnv, self).__init__(config)
    # self.blanket_id = -1
    self.blanket_pos = [0, 0, 0]
    self.blanket_rz = 0
    self.blanket_size = [0.2, 0.15, 0.002]
    self.blanket = Blanket()
    self.place_offset = 0.04

  def resetBlanket(self):
    self.blanket_rz = -np.pi/2 + np.random.random_sample() * 2 * np.pi
    self.blanket_pos = self._getValidPositions(0.2, 0, [], 1)[0]
    self.blanket_pos.append(self.blanket_size[2]/2)
    if self.blanket.id is not None:
      self.blanket.remove()
    self.blanket.initialize(pos=self.blanket_pos, rot=pb.getQuaternionFromEuler((0, 0, self.blanket_rz)), size=self.blanket_size)

  def _getExistingXYPositions(self):
    positions = [o.getXYPosition() for o in self.objects]
    positions.append(self.blanket_pos[:2])
    return positions

  def reset(self):
    ''''''
    while True:
      self.resetPybulletEnv()
      self.resetBlanket()
      try:
        self._generateShapes(constants.BOWL, 1, random_orientation=self.random_orientation, min_distance=5*self.max_block_size)
        self._generateShapes(constants.SPOON, 1, random_orientation=self.random_orientation, min_distance=5*self.max_block_size)
        self._generateShapes(constants.CUP, 1, random_orientation=self.random_orientation, min_distance=5*self.max_block_size)
      except NoValidPositionException as e:
        continue
      else:
        break
    return self._getObservation()

  def _checkTermination(self):
    ''''''
    bowl_xy = self.objects[0].getXYPosition()
    spoon_xy = self.objects[1].getXYPosition()
    cup_xy = self.objects[2].getXYPosition()
    theta = np.arctan((bowl_xy[1] - cup_xy[1]) / (bowl_xy[0] - cup_xy[0] + 1e-5))
    if theta < 0:
      theta += np.pi
    angle_diff = abs(theta - self.blanket_rz)
    angle_diff = min(angle_diff, abs(angle_diff - np.pi))

    angle_right = angle_diff < np.pi/10
    bowl_spoon_close = np.linalg.norm(np.array(bowl_xy)-spoon_xy) < 0.05
    spoon_touch_bowl = self.objects[0].isTouching(self.objects[1])
    bowl_touch_blanket = self.objects[0].isTouching(self.blanket)
    cup_touch_blanket = self.objects[2].isTouching(self.blanket)
    bowl_upright = self._checkObjUpright(self.objects[0])
    cup_upright = self._checkObjUpright(self.objects[2])

    return all([angle_right, bowl_spoon_close, spoon_touch_bowl, bowl_touch_blanket, cup_touch_blanket, bowl_upright, cup_upright])

def createBowlSpoonCupEnv(config):
  return BowlSpoonCupEnv(config)
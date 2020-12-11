import numpy as np
import pybullet as pb
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.constants import NoValidPositionException
from helping_hands_rl_envs.planners.base_planner import BasePlanner

class DrawerTeapotPlanner(BasePlanner):
  def __init__(self, env, config):
    super().__init__(env, config)

  def openDrawer(self, drawer):
    handle1_pos = drawer.getHandlePosition()
    rot = (0, -np.pi / 2, 0)
    return self.encodeAction(constants.PULL_PRIMATIVE, handle1_pos[0], handle1_pos[1], handle1_pos[2], rot)

  def pickTeapot(self):
    pos = self.env.objects[0].getHandlePos()
    rx, ry, rz = list(pb.getEulerFromQuaternion(self.env.objects[0].getRotation()))
    return self.encodeAction(constants.PICK_PRIMATIVE, pos[0], pos[1], pos[2], (rz, ry, rx))

  def pickTeapotLid(self):
    pos = self.env.objects[1].getPosition()
    rx, ry, rz = list(pb.getEulerFromQuaternion(self.env.objects[1].getRotation()))
    return self.encodeAction(constants.PICK_PRIMATIVE, pos[0], pos[1], pos[2], (rz, ry, rx))

  def placeTeapotOnGround(self):
    sample_range = [[0.33, 0.37],
                    [-0.15, 0.15]]
    try:
      place_pos = self.getValidPositions(0.05, 0.05, [self.env.objects[1].getPosition()[:2]], 1, sample_range)[0]
    except NoValidPositionException:
      place_pos = self.getValidPositions(0.05, 0.05, [], 1, sample_range)[0]
    x, y, z = place_pos[0], place_pos[1], self.env.place_offset
    r = 0
    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)

  def placeLidOnGround(self):
    sample_range = [[0.33, 0.37],
                    [-0.15, 0.15]]
    try:
      place_pos = self.getValidPositions(0.05, 0.05, [self.env.objects[0].getPosition()[:2]], 1, sample_range)[0]
    except NoValidPositionException:
      place_pos = self.getValidPositions(0.05, 0.05, [], 1, sample_range)[0]
    x, y, z = place_pos[0], place_pos[1], self.env.place_offset
    r = 0
    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)

  def placeLidOnTeapot(self):
    teapot_open_pos = self.env.objects[0].getOpenPos()
    x, y, z = teapot_open_pos
    r = 0
    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)

  def getNextAction(self):
    teapot = self.env.objects[0]
    lid = self.env.objects[1]
    teapot_in_drawer1 = self.env.drawer1.isObjInsideDrawer(teapot)
    teapot_in_drawer2 = self.env.drawer2.isObjInsideDrawer(teapot)
    lid_in_drawer1 = self.env.drawer1.isObjInsideDrawer(lid)
    lid_in_drawer2 = self.env.drawer2.isObjInsideDrawer(lid)
    teapot_in_drawer = teapot_in_drawer1 or teapot_in_drawer2
    lid_in_drawer = lid_in_drawer1 or lid_in_drawer2
    drawer1_open = self.env.drawer1.isDrawerOpen()
    drawer2_open = self.env.drawer2.isDrawerOpen()

    # if holding object, do place
    if self.isHolding():
      if self.isObjectHeld(teapot):
        # if holding teapot, place on ground
        return self.placeTeapotOnGround()
      elif self.isObjectHeld(lid):
        if teapot_in_drawer:
          # if holding lid but teapot still in drawer, place on ground
          return self.placeLidOnGround()
        else:
          # if holding lid and teapot out, place on teapot
          return self.placeLidOnTeapot()
    else:
      if teapot_in_drawer or lid_in_drawer:
        if not drawer1_open:
          # if teapot or lid in drawer and drawer1 closed, open drawer1
          return self.openDrawer(self.env.drawer1)
        elif drawer1_open and (teapot_in_drawer1 or lid_in_drawer1):
          if teapot_in_drawer1:
            # if drawer1 opened and teapot in drawer1, pick teapot
            return self.pickTeapot()
          else:
            # if drawer1 opened and lid in drawer1, pick lid
            return self.pickTeapotLid()
        # drawer1 clear

        elif not drawer2_open:
          return self.openDrawer(self.env.drawer2)
        else:
          if teapot_in_drawer2:
            return self.pickTeapot()
          else:
            return self.pickTeapotLid()
      else:
        return self.pickTeapotLid()







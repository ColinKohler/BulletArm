import pybullet as pb
import numpy as np
import scipy
from bulletarm.pybullet.utils.sensor import Sensor
from bulletarm.pybullet.utils import transformations

class Gelsight(object):
  def __init__(self):
    pass

  def getFingerImg(self, finger_a_pos, finger_b_pos):
    cam_up_vector = [0, 0, 1]

    cam_a_pos = finger_a_pos
    cam_a_target_pos = [finger_a_pos[0], finger_a_pos[1], finger_a_pos[2]]
    far_a = np.linalg.norm(np.array(cam_a_pos) - np.array(cam_a_target_pos)) + 2
    sensor_a = Sensor(cam_1_pos, cam_up_vector, cam_target_pos, 2, near=0.01, far=far_1)

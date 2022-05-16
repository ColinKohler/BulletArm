import pybullet as pb
import numpy as np
from bulletarm.pybullet.utils.sensor import Sensor

class OrthographicSensor(Sensor):
  def __init__(self, cam_pos, cam_up_vector, target_pos, target_size, near, far):
    super().__init__(cam_pos, cam_up_vector, target_pos, target_size, near, far)
    self.target_size = target_size
    self.proj_matrix = np.array([
      [2 / target_size, 0, 0, 0],
      [0, 2 / target_size, 0, 0],
      [0, 0, -2/(self.far - self.near), -(self.far+self.near)/(self.far-self.near)],
      [0, 0, 0, 1]
    ]).T.reshape(16)

  def setCamMatrix(self, cam_pos, cam_up_vector, target_pos):
    self.view_matrix = pb.computeViewMatrix(
      cameraEyePosition=[cam_pos[0], cam_pos[1], 1],
      cameraUpVector=cam_up_vector,
      cameraTargetPosition=target_pos,
    )
    self.near = 1-cam_pos[2]
    self.proj_matrix = np.array([
      [2 / self.target_size, 0, 0, 0.],
      [0, 2 / self.target_size, 0, 0.],
      [0, 0, -2/(self.far - self.near), -(self.far+self.near)/(self.far-self.near)],
      [0, 0, 0, 1]
    ]).T.reshape(16)
    # depth = self.getDepth(128)

  def getDepth(self, size):
    image_arr = pb.getCameraImage(width=size, height=size,
                                  viewMatrix=self.view_matrix,
                                  projectionMatrix=self.proj_matrix)
    depth = np.array(image_arr[3])
    depth = self.far * self.near / (self.far - (self.far - self.near) * depth)

    return depth

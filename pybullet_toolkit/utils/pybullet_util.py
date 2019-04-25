import pybullet as pb
import numpy as np

def getMatrix(pos, rot):
  T = np.eye(4)
  T[:3, :3] = np.array(pb.getMatrixFromQuaternion(rot)).reshape((3, 3))
  T[:3, 3] = pos
  return T
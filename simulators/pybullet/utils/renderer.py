import pybullet as pb
import numpy as np
import open3d as o3d
import trimesh
import pyrender
import matplotlib.pyplot as plt

class Renderer(object):
  def __init__(self):
    self.points = np.empty((0, 3))

  def addPoints(self, points):
    self.points = np.concatenate((self.points, points))

  def clearPoints(self):
    self.points = np.empty((0, 3))

  def renderHeightmapPerspective(self, size, cam_pos, cam_up_vector, target_pos, target_size, near, far):
    view_matrix = pb.computeViewMatrix(
      cameraEyePosition=cam_pos,
      cameraUpVector=cam_up_vector,
      cameraTargetPosition=target_pos,
    )
    view_matrix = np.asarray(view_matrix).reshape([4, 4], order='F')
    # fov in rad
    fov = 2 * np.arctan((target_size / 2) / far)
    camera_pose = np.linalg.inv(view_matrix)

    mesh = pyrender.Mesh.from_points(self.points)
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.PerspectiveCamera(fov, znear=near, zfar=far, aspectRatio=1)
    # camera = pyrender.OrthographicCamera(xmag=0.2, ymag=0.2, znear=near, zfar=far)
    scene.add(camera, pose=camera_pose)
    # pyrender.Viewer(scene)
    r = pyrender.OffscreenRenderer(size, size)
    color, depth = r.render(scene)
    depth[depth==0] = far
    return np.abs(depth - np.max(depth))

  def renderHeightmapOrthographic(self, size, cam_pos, cam_up_vector, target_pos, target_size, near, far):
    view_matrix = pb.computeViewMatrix(
      cameraEyePosition=cam_pos,
      cameraUpVector=cam_up_vector,
      cameraTargetPosition=target_pos,
    )
    view_matrix = np.asarray(view_matrix).reshape([4, 4], order='F')
    camera_pose = np.linalg.inv(view_matrix)

    mesh = pyrender.Mesh.from_points(self.points)
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.OrthographicCamera(xmag=target_size/2, ymag=target_size/2, znear=near, zfar=far)
    scene.add(camera, pose=camera_pose)
    pyrender.Viewer(scene)
    r = pyrender.OffscreenRenderer(size, size)
    color, depth = r.render(scene)
    depth[depth==0] = far
    return np.abs(depth - np.max(depth))
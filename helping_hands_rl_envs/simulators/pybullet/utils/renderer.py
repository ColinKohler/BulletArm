import time
import pybullet as pb
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from helping_hands_rl_envs.simulators.pybullet.utils.sensor import Sensor

class Renderer(object):
  def __init__(self, workspace):
    self.workspace = workspace

    cam_forward_target_pos = [0.8, self.workspace[1].mean(), 0]
    cam_forward_up_vector = [0, 0, 1]

    cam_1_forward_pos = [0, 0.5, 1]
    far_1 = np.linalg.norm(np.array(cam_1_forward_pos) - np.array(cam_forward_target_pos)) + 2
    self.sensor_1 = Sensor(cam_1_forward_pos, cam_forward_up_vector, cam_forward_target_pos,
                           3.2, near=0.5, far=far_1)

    cam_2_forward_pos = [0, -0.5, 1]
    far_2 = np.linalg.norm(np.array(cam_2_forward_pos) - np.array(cam_forward_target_pos)) + 2
    self.sensor_2 = Sensor(cam_2_forward_pos, cam_forward_up_vector, cam_forward_target_pos,
                           3.2, near=0.5, far=far_2)

    cam_3_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 20]
    cam_3_target_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 0]
    far_3 = np.linalg.norm(np.array(cam_3_pos) - np.array(cam_3_target_pos)) + 2
    self.sensor_3 = Sensor(cam_3_pos, [-1, 0, 0], cam_3_target_pos,
                           self.workspace[0][1] - self.workspace[0][0], near=0.1, far=far_3)

    self.points = cp.empty((0, 3))

  def getNewPointCloud(self):
    points1 = self.sensor_1.getPointCloud(480, to_numpy=False)
    points2 = self.sensor_2.getPointCloud(480, to_numpy=False)
    points3 = self.sensor_3.getPointCloud(128, to_numpy=False)
    self.clearPoints()
    self.addPoints(points1)
    self.addPoints(points2)
    self.addPoints(points3)

  def getForwardHeightmapByThetas(self, size, thetas):
    heightmaps = []
    for theta in thetas:
      dy = np.sin(theta) * 1
      dx = np.cos(theta) * 1

      render_cam_target_pos = [self.workspace[0].mean() + 0.41, self.workspace[1].mean(), self.workspace[2].mean()]
      render_cam_up_vector = [0, 0, 1]

      render_cam_pos1 = [self.workspace[0].mean() + 0.41-dx, -dy, self.workspace[2].mean()]
      hm = self.projectHeightmap(size, render_cam_pos1, render_cam_up_vector,
                                 render_cam_target_pos, self.workspace[2][1] - self.workspace[2][0])

      heightmaps.append(hm)
    heightmaps = np.stack(heightmaps)
    return heightmaps

  def getTopDownHeightmap(self, size):
    render_cam_target_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 0]
    render_cam_up_vector = [-1, 0, 0]

    render_cam_pos1 = [self.workspace[0].mean(), self.workspace[1].mean(), 10]
    # t0 = time.time()
    hm = self.projectHeightmap(size, render_cam_pos1, render_cam_up_vector,
                               render_cam_target_pos, self.workspace[2][1] - self.workspace[2][0])
    return hm

  def addPoints(self, points):
    self.points = cp.concatenate((self.points, points))

  def clearPoints(self):
    self.points = cp.empty((0, 3))

  def projectHeightmap(self, size, cam_pos, cam_up_vector, target_pos, target_size):
    view_matrix = pb.computeViewMatrix(
      cameraEyePosition=cam_pos,
      cameraUpVector=cam_up_vector,
      cameraTargetPosition=target_pos,
    )
    view_matrix = cp.asarray(view_matrix).reshape([4, 4], order='F')

    augment = cp.ones((1, self.points.shape[0]))
    # pts = cp.concatenate((cp.asarray(self.points).T, augment), axis=0)
    pts = cp.concatenate((self.points.T, augment), axis=0)
    projection_matrix = cp.array([
      [1 / (target_size / 2), 0, 0, 0],
      [0, 1 / (target_size / 2), 0, 0],
      [0, 0, -1, 0],
      [0, 0, 0, 1]
    ])
    tran_world_pix = cp.matmul(projection_matrix, view_matrix)
    pts = cp.matmul(tran_world_pix, pts)
    pts[1] = -pts[1]
    pts[0] = (pts[0] + 1) * size / 2
    pts[1] = (pts[1] + 1) * size / 2

    # pts_floor = pts.copy()
    # pts_floor[0], pts_floor[1] = cp.floor(pts_floor[0]), cp.floor(pts_floor[1])
    # pts_ceil = pts.copy()
    # pts_ceil[0], pts_ceil[1] = cp.ceil(pts_ceil[0]), cp.ceil(pts_ceil[1])
    # pts = cp.concatenate((pts_floor, pts_ceil), 1)
    # mask = (pts[0] >= 0) * (pts[0] < size) * (pts[1] >= 0) * (pts[1] < size)

    pts[0] = cp.round_(pts[0])
    pts[1] = cp.round_(pts[1])
    mask = (pts[0] >= 0) * (pts[0] < size) * (pts[1] > 0) * (pts[1] < size)
    pts = pts[:, mask]
    # dense pixel index
    mix_xy = (pts[1].astype(int) * size + pts[0].astype(int))
    # lexsort point cloud first on dense pixel index, then on z value
    ind = cp.lexsort(cp.stack((pts[2], mix_xy)))
    # bin count the points that belongs to each pixel
    bincount = cp.bincount(mix_xy)
    # cumulative sum of the bin count. the result indicates the cumulative sum of number of points for all previous pixels
    cumsum = cp.cumsum(bincount)
    # rolling the cumsum gives the ind of the first point that belongs to each pixel.
    # because of the lexsort, the first point has the smallest z value
    cumsum = cp.roll(cumsum, 1)
    cumsum[0] = bincount[0]
    # pad for unobserved pixels
    cumsum = cp.concatenate((cumsum, -1 * cp.ones(size * size - cumsum.shape[0]))).astype(int)

    depth = pts[2][ind][cumsum]
    depth[cumsum == 0] = cp.nan
    depth = depth.reshape(size, size)
    depth = cp.asnumpy(depth)
    mask = np.isnan(depth)
    depth[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), depth[~mask])

    return np.abs(depth - np.max(depth))


  # def projectHeightmap(self, size, cam_pos, cam_up_vector, target_pos, target_size):
  #   view_matrix = pb.computeViewMatrix(
  #     cameraEyePosition=cam_pos,
  #     cameraUpVector=cam_up_vector,
  #     cameraTargetPosition=target_pos,
  #   )
  #   view_matrix = np.asarray(view_matrix).reshape([4, 4], order='F')
  #
  #   augment = np.ones((1, self.points.shape[0]))
  #   pts = np.concatenate((self.points.T, augment), axis=0)
  #   projection_matrix = np.array([
  #     [1 / (target_size / 2), 0, 0, 0],
  #     [0, 1 / (target_size / 2), 0, 0],
  #     [0, 0, -1, 0],
  #     [0, 0, 0, 1]
  #   ])
  #   tran_world_pix = np.matmul(projection_matrix, view_matrix)
  #   pts = np.matmul(tran_world_pix, pts)
  #   pts[1] = -pts[1]
  #   pts[0] = (pts[0] + 1) * size / 2
  #   pts[1] = (pts[1] + 1) * size / 2
  #   mask = (pts[0] > 0) * (pts[0] < size) * (pts[1] > 0) * (pts[1] < size)
  #   pts = pts[:, mask]
  #   depth = np.ones((size, size)) * 1000
  #   for i in range(pts.shape[1]):
  #     depth[int(pts[1, i]), int(pts[0, i])] = min(depth[int(pts[1, i]), int(pts[0, i])], pts[2, i])
  #   depth[depth == 1000] = np.nan
  #   mask = np.isnan(depth)
  #   depth[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), depth[~mask])
  #
  #   return np.abs(depth - np.max(depth))
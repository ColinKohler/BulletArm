import time
import pybullet as pb
import numpy as np
import cupy as cp
import scipy
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from helping_hands_rl_envs.simulators.pybullet.utils.sensor import Sensor
import skimage.transform as sk_transform
from helping_hands_rl_envs.simulators.pybullet.utils import transformations

class Renderer(object):
  def __init__(self, workspace):
    self.workspace = workspace

    cam_forward_target_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 0]
    cam_forward_up_vector = [0, 0, 1]

    cam_1_forward_pos = [self.workspace[0].mean(), 0.5, 1]
    far_1 = np.linalg.norm(np.array(cam_1_forward_pos) - np.array(cam_forward_target_pos)) + 2
    self.sensor_1 = Sensor(cam_1_forward_pos, cam_forward_up_vector, cam_forward_target_pos,
                           2, near=0.1, far=far_1)

    cam_2_forward_pos = [self.workspace[0].mean(), -0.5, 1]
    far_2 = np.linalg.norm(np.array(cam_2_forward_pos) - np.array(cam_forward_target_pos)) + 2
    self.sensor_2 = Sensor(cam_2_forward_pos, cam_forward_up_vector, cam_forward_target_pos,
                           2, near=0.1, far=far_2)


    self.points = cp.empty((0, 3))

  def getNewPointCloud(self, res=256):
    self.clearPoints()
    # ceiling = np.array(np.meshgrid(np.linspace(self.workspace[0][0], self.workspace[0][1], 256),
    #                                np.linspace(self.workspace[1][0], self.workspace[1][1], 256))).T.reshape(-1, 2)
    # ceiling = np.concatenate((ceiling, 0.25 * np.ones((256*256, 1))), 1)
    # self.addPoints(cp.array(ceiling))
    points1 = self.sensor_1.getPointCloud(res, to_numpy=False)
    points2 = self.sensor_2.getPointCloud(res, to_numpy=False)
    self.addPoints(points1)
    self.addPoints(points2)
    self.points = self.points[self.points[:, 2] <= self.workspace[2][1]]
    # import pyrender
    # mesh = pyrender.Mesh.from_points(self.points.get())
    # scene = pyrender.Scene()
    # scene.add(mesh)
    # pyrender.Viewer(scene)

  def getTopDownDepth(self, target_size, img_size, gripper_pos, gripper_rz):
    if self.points.shape[0] == 0:
      self.getNewPointCloud(512)
      self.points = self.points[self.points[:, 2] <= max(gripper_pos[2]-0.01, 0.05)]
    # self.points = self.points[(self.workspace[0, 0] <= self.points[:, 0]) * (self.points[:, 0] <= self.workspace[0, 1])]
    # self.points = self.points[(self.workspace[1, 0] <= self.points[:, 1]) * (self.points[:, 1] <= self.workspace[1, 1])]

    render_cam_target_pos = [gripper_pos[0], gripper_pos[1], 0]
    # render_cam_up_vector = [-1, 0, 0]
    T = transformations.euler_matrix(0, 0, gripper_rz)
    render_cam_up_vector = T.dot(np.array([-1, 0, 0, 1]))[:3]


    render_cam_pos1 = [gripper_pos[0], gripper_pos[1], gripper_pos[2]]
    # t0 = time.time()
    depth = self.projectDepth(img_size, render_cam_pos1, render_cam_up_vector,
                               render_cam_target_pos, target_size)
    # depth = sk_transform.rotate(depth, np.rad2deg(gripper_rz))
    return depth


  def getTopDownHeightmap(self, size):
    render_cam_target_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 0]
    render_cam_up_vector = [-1, 0, 0]

    render_cam_pos1 = [self.workspace[0].mean(), self.workspace[1].mean(), 10]
    # t0 = time.time()
    hm = self.projectHeightmap(size, render_cam_pos1, render_cam_up_vector,
                               render_cam_target_pos, self.workspace[0][1] - self.workspace[0][0])
    return hm

  def addPoints(self, points):
    self.points = cp.concatenate((self.points, points))

  def clearPoints(self):
    self.points = cp.empty((0, 3))

  def interpolate(self, depth):
    # a boolean array of (width, height) which False where there are missing values and True where there are valid (non-missing) values
    mask = np.logical_not(np.isnan(depth))
    # array of (number of points, 2) containing the x,y coordinates of the valid values only
    xx, yy = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    xym = np.vstack((np.ravel(xx[mask]), np.ravel(yy[mask]))).T

    # the valid values in the first, second, third color channel,  as 1D arrays (in the same order as their coordinates in xym)
    data0 = np.ravel(depth[:, :][mask])

    # three separate interpolators for the separate color channels
    interp0 = scipy.interpolate.NearestNDInterpolator(xym, data0)

    # interpolate the whole image, one color channel at a time
    result0 = interp0(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)

    return result0

  def projectDepth(self, size, cam_pos, cam_up_vector, target_pos, target_size):
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
    cumsum[cumsum == cp.roll(cumsum, -1)] = 0
    # pad for unobserved pixels
    cumsum = cp.concatenate((cumsum, -1 * cp.ones(size * size - cumsum.shape[0]))).astype(int)

    depth = pts[2][ind][cumsum]
    depth[cumsum == 0] = cp.nan
    depth = depth.reshape(size, size)
    depth = cp.asnumpy(depth)

    depth = self.interpolate(depth)

    # mask = np.isnan(depth)
    # depth[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), depth[~mask])

    # imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    # depth = imputer.fit_transform(depth)
    # if depth.shape != (size, size):
    #   depth = sk_transform.resize(depth, (size, size), mode='constant', anti_aliasing=0)

    return depth

  def projectHeightmap(self, size, cam_pos, cam_up_vector, target_pos, target_size):
    depth = self.projectDepth(size, cam_pos, cam_up_vector, target_pos, target_size)
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
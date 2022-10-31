import pybullet as pb
import numpy as np
import numpy.random as npr
import scipy
from bulletarm.pybullet.utils.sensor import Sensor
from bulletarm.pybullet.utils import transformations

class Renderer(object):
  def __init__(self, workspace):
    self.workspace = workspace

    self.cam_forward_target_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 0]
    self.cam_forward_up_vector = [0, 0, 1]

    self.cam_1_forward_pos = [self.workspace[0].mean(), 0.5, 1]
    far_1 = np.linalg.norm(np.array(self.cam_1_forward_pos) - np.array(self.cam_forward_target_pos)) + 2
    self.sensor_1 = Sensor(self.cam_1_forward_pos, self.cam_forward_up_vector, self.cam_forward_target_pos,
                           2, near=0.1, far=far_1)

    self.cam_2_forward_pos = [self.workspace[0].mean(), -0.5, 1]
    far_2 = np.linalg.norm(np.array(self.cam_2_forward_pos) - np.array(self.cam_forward_target_pos)) + 2
    self.sensor_2 = Sensor(self.cam_2_forward_pos, self.cam_forward_up_vector, self.cam_forward_target_pos,
                           2, near=0.1, far=far_2)


    self.points = np.empty((0, 3))

  def gitterSensors(self):
    cam_1_forward_pos = np.array(self.cam_1_forward_pos) + npr.uniform(-0.1, 0.1, 3)
    self.sensor_1.setCamMatrix(cam_1_forward_pos, self.cam_forward_up_vector, self.cam_forward_target_pos)

    cam_2_forward_pos = np.array(self.cam_2_forward_pos) + npr.uniform(-0.1, 0.1, 3)
    self.sensor_1.setCamMatrix(cam_2_forward_pos, self.cam_forward_up_vector, self.cam_forward_target_pos)

  def getNewPointCloud(self, res=256):
    self.clearPoints()
    # ceiling = np.array(np.meshgrid(np.linspace(self.workspace[0][0], self.workspace[0][1], 256),
    #                                np.linspace(self.workspace[1][0], self.workspace[1][1], 256))).T.reshape(-1, 2)
    # ceiling = np.concatenate((ceiling, 0.25 * np.ones((256*256, 1))), 1)
    # self.addPoints(np.array(ceiling))
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
    points = np.copy(self.points)
    points = points[points[:, 2] <= max(gripper_pos[2]-0.01, 0.05)]
    # self.points = self.points[(self.workspace[0, 0] <= self.points[:, 0]) * (self.points[:, 0] <= self.workspace[0, 1])]
    # self.points = self.points[(self.workspace[1, 0] <= self.points[:, 1]) * (self.points[:, 1] <= self.workspace[1, 1])]

    render_cam_target_pos = [gripper_pos[0], gripper_pos[1], 0]
    # render_cam_up_vector = [-1, 0, 0]
    T = transformations.euler_matrix(0, 0, gripper_rz)
    render_cam_up_vector = T.dot(np.array([-1, 0, 0, 1]))[:3]


    render_cam_pos1 = [gripper_pos[0], gripper_pos[1], gripper_pos[2]]
    # t0 = time.time()
    depth = self.projectDepth(points, img_size, render_cam_pos1, render_cam_up_vector,
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
    self.points = np.concatenate((self.points, points))

  def clearPoints(self):
    self.points = np.empty((0, 3))

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

  def projectDepth(self, points, size, cam_pos, cam_up_vector, target_pos, target_size):
    view_matrix = pb.computeViewMatrix(
      cameraEyePosition=cam_pos,
      cameraUpVector=cam_up_vector,
      cameraTargetPosition=target_pos,
    )
    view_matrix = np.asarray(view_matrix).reshape([4, 4], order='F')

    augment = np.ones((1, points.shape[0]))
    # pts = np.concatenate((np.asarray(points).T, augment), axis=0)
    pts = np.concatenate((points.T, augment), axis=0)
    projection_matrix = np.array([
      [1 / (target_size / 2), 0, 0, 0],
      [0, 1 / (target_size / 2), 0, 0],
      [0, 0, -1, 0],
      [0, 0, 0, 1]
    ])
    tran_world_pix = np.matmul(projection_matrix, view_matrix)
    pts = np.matmul(tran_world_pix, pts)
    pts[1] = -pts[1]
    pts[0] = (pts[0] + 1) * size / 2
    pts[1] = (pts[1] + 1) * size / 2

    pts[0] = np.round_(pts[0])
    pts[1] = np.round_(pts[1])
    mask = (pts[0] >= 0) * (pts[0] < size) * (pts[1] > 0) * (pts[1] < size)
    pts = pts[:, mask]
    if pts.shape[1] == 0:
      return np.zeros([size, size])
    # dense pixel index
    mix_xy = (pts[1].astype(int) * size + pts[0].astype(int))
    # lexsort point cloud first on dense pixel index, then on z value
    ind = np.lexsort(np.stack((pts[2], mix_xy)))
    # bin count the points that belongs to each pixel
    bincount = np.bincount(mix_xy)
    # cumulative sum of the bin count. the result indicates the cumulative sum of number of points for all previous pixels
    cumsum = np.cumsum(bincount)
    # rolling the cumsum gives the ind of the first point that belongs to each pixel.
    # because of the lexsort, the first point has the smallest z value
    cumsum = np.roll(cumsum, 1)
    cumsum[0] = bincount[0]
    cumsum[cumsum == np.roll(cumsum, -1)] = 0
    # pad for unobserved pixels
    cumsum = np.concatenate((cumsum, -1 * np.ones(size * size - cumsum.shape[0]))).astype(int)

    depth = pts[2][ind][cumsum]
    depth[cumsum == 0] = np.nan
    depth = depth.reshape(size, size)
    depth = self.interpolate(depth)

    return depth

  def projectHeightmap(self, size, cam_pos, cam_up_vector, target_pos, target_size):
    depth = self.projectDepth(self.points, size, cam_pos, cam_up_vector, target_pos, target_size)
    return np.abs(depth - np.max(depth))

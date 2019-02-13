import sys
import numpy as np
import skimage

from vrep_arm_toolkit.simulation import vrep
import vrep_arm_toolkit.utils.vrep_utils as utils
import vrep_arm_toolkit.utils.transformations as transformations

VREP_BLOCKING = vrep.simx_opmode_blocking

class VisionSensor(object):
  def __init__(self, sim_client, sensor_name,
                     workspace, intrinsics,
                     get_rgb=True, get_depth=True,
                     z_near=0.01, z_far=10):
    '''
    VRep vision sensor class.

    Args:
      - sim_client: VRep client object to communicate with simulator over
      - sensor_name: Sensor name in simulator
      - workspace: Workspace for the vision sensor in robot coordinates
      - intrinsics: sensor intrinsics
      - get_rgb: Should the sensor get the rgb image
      - get_depth: Should the sensor get the depth image
      - z_near: Minimum distance for depth sensing
      - z_far: Maximum distance for depth sensing
    '''
    self.sim_client = sim_client
    self.workspace = workspace
    self.intrinsics = intrinsics
    self.get_rgb = get_rgb
    self.get_depth = get_depth

    # Setup sensor and default sensor values
    sim_ret, self.sensor = utils.getObjectHandle(self.sim_client, sensor_name)
    sim_ret, pos = utils.getObjectPosition(self.sim_client, self.sensor)
    sim_ret, rot = utils.getObjectOrientation(self.sim_client, self.sensor)
    cam_trans = np.eye(4,4)
    cam_trans[:3,3] = np.asarray(pos)
    rotm = np.linalg.inv(transformations.euler_matrix(-rot[0], -rot[1], -rot[2]))
    self.pose = np.dot(cam_trans, rotm)

    self.z_near = z_near
    self.z_far = z_far

  def getData(self, use_float=False):
    '''
    Get the RGB-D data from the sensor

    Returns: (rgb, depth)
      - rgb: (resolution, resolution, 3) RGB image
      - depth: (resolution, resolution) depth image
    '''
    if not self.get_rgb or not self.get_depth:
      print('Either depth or rgb data is not enabled.')
      return None

    return self.getDepthData(), self.getColorData(use_float=use_float)

  def getDepthData(self):
    '''
    Get the depth data from the sensor.

    Returns: (resolution, resolution) depth image
    '''
    if not self.get_rgb:
      print('Cannot get depth data without depth enabled')
      return None

    sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client, self.sensor, VREP_BLOCKING)
    depth_img = np.asarray(depth_buffer)
    depth_img.shape = (resolution[1], resolution[0])
    depth_img = depth_img * (self.z_far - self.z_near) + self.z_near

    return depth_img

  def getColorData(self, use_float=False):
    '''
    Get the RGB data from the sensor.

    Returns: (resolution, resolution, 3) RGB image
    '''
    if not self.get_rgb:
      print('Cannot get color data without rgb enabled')
      return None

    sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.sensor, 0, VREP_BLOCKING)
    color_img = np.asarray(raw_image)
    color_img.shape = (resolution[1], resolution[0], 3)
    color_img = color_img.astype(np.float) / 255
    color_img[color_img < 0] += 1
    color_img *= 255
    color_img = np.fliplr(color_img)
    color_img = color_img.astype(np.uint8)

    if use_float:
      color_img = skimage.img_as_float(color_img)

    return color_img

  def getPointCloud(self):
    '''
    Get point cloud from the sensor. If RGB is not enabled just the depth points will be returned.
    Depth must be enabled to use this method.

    Returns: Tuple of (depth_points, rgb_points)
    '''
    if not self.get_depth:
      print('Cannot get point cloud without depth enabled')
      return None

    depth_img = self.getDepthData()
    depth_img = np.fliplr(depth_img)
    depth_h, depth_w = depth_img.shape

    # Project depth into 3D point cloud in camera coordinates
    pix_x, pix_y = np.meshgrid(np.linspace(0, depth_w-1, depth_w),
                               np.linspace(0, depth_h-1, depth_h))
    cam_pts_x = np.multiply(pix_x - self.intrinsics[0,2], depth_img / self.intrinsics[0,0])
    cam_pts_y = np.multiply(pix_y - self.intrinsics[1,2], depth_img / self.intrinsics[1,1])
    cam_pts_z = np.copy(depth_img)
    cam_pts_x = cam_pts_x.reshape(depth_h  * depth_w, 1)
    cam_pts_y = cam_pts_y.reshape(depth_h  * depth_w, 1)
    cam_pts_z = cam_pts_z.reshape(depth_h  * depth_w, 1)

    cam_pts = np.hstack((cam_pts_x, cam_pts_y, cam_pts_z))

    # Get the rgb values for the various points
    rgb_pts = None
    if self.get_rgb:
      color_img = self.getColorData(use_float=True)
      rgb_pts_r = color_img[:,:,0]
      rgb_pts_g = color_img[:,:,1]
      rgb_pts_b = color_img[:,:,2]
      rgb_pts_r = rgb_pts_r.reshape(depth_h  * depth_w, 1)
      rgb_pts_g = rgb_pts_g.reshape(depth_h  * depth_w, 1)
      rgb_pts_b = rgb_pts_b.reshape(depth_h  * depth_w, 1)

      rgb_pts = np.hstack((rgb_pts_r, rgb_pts_g, rgb_pts_b))

    # Transform the point cloud to the camera pose
    cam_pts = np.transpose(np.dot(self.pose[:3,:3], np.transpose(cam_pts)) + \
                           np.tile(self.pose[:3,3:], (1, cam_pts.shape[0])))

    # Filter out points outside workspace
    valid_x = (cam_pts[:,0] >= self.workspace[0,0]) & \
              (cam_pts[:,0] <= self.workspace[0,1])
    valid_y = (cam_pts[:,1] >= self.workspace[1,0]) & \
              (cam_pts[:,1] <= self.workspace[1,1])
    valid_z = (cam_pts[:,2] >= self.workspace[2,0]) & \
              (cam_pts[:,2] <= self.workspace[2,1])
    valid = np.where(valid_x & valid_y & valid_z)
    cam_pts = cam_pts[valid]
    rgb_pts = rgb_pts[valid]

    return cam_pts, rgb_pts

  def getHeightmap(self, resolution=0.002):
    '''
    Get depth and color heightmaps from the sensor.

    Args:
      - resolution: Resolution of the heightmap (meters per pixel)
    '''
    # Create height map from workspace
    heightmap_size = np.round(((self.workspace[1,1] - self.workspace[1,0]) / resolution,
                               (self.workspace[0,1] - self.workspace[0,0]) / resolution)).astype(int)

    # Get point cloud and init heightmaps
    depth_pts, rgb_pts = self.getPointCloud()
    color_heightmap_r = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.float32)
    color_heightmap_g = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.float32)
    color_heightmap_b = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.float32)
    depth_heightmap = np.zeros(heightmap_size)

    heightmap_pix_x = np.floor((depth_pts[:,0] - self.workspace[0,0]) / resolution).astype(int)
    heightmap_pix_y = np.floor((depth_pts[:,1] - self.workspace[1,0]) / resolution).astype(int)

    # Load data into height maps and post-processes a bit
    color_heightmap_r[heightmap_pix_y, heightmap_pix_x] = rgb_pts[:,[0]]
    color_heightmap_g[heightmap_pix_y, heightmap_pix_x] = rgb_pts[:,[1]]
    color_heightmap_b[heightmap_pix_y, heightmap_pix_x] = rgb_pts[:,[2]]
    color_heightmap = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)

    depth_heightmap[heightmap_pix_y, heightmap_pix_x] = depth_pts[:,2]
    depth_heightmap -= self.workspace[2,0]
    depth_heightmap[depth_heightmap < 0] = 0
    depth_heightmap[depth_heightmap == -self.workspace[2,0]] = 0

    return depth_heightmap, color_heightmap

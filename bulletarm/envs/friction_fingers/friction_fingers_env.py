import pybullet as pb
import numpy as np
import numpy.random as npr

from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils import transformations
from bulletarm.pybullet.utils.renderer import Renderer
from bulletarm.pybullet.utils.ortho_sensor import OrthographicSensor
from bulletarm.pybullet.utils.sensor import Sensor

class FrictionFingersEnv(BaseEnv):
  def __init__(self, config):
    # Only allow the robot with the Yale openhand VF gripper
    config['robot'] = 'kuka'

    super().__init__(config)

    self.renderer = None
    self.pers_sensor = None
    self.obs_size_m = self.workspace_size * self.view_scale
    self.initSensor()

  def initialize(self):
    super().initialize()

  def initSensor(self):
    cam_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 0.29]
    target_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 0]
    cam_up_vector = [-1, 0, 0]
    self.sensor = OrthographicSensor(cam_pos, cam_up_vector, target_pos, self.obs_size_m, 0.1, 1)
    self.sensor.setCamMatrix(cam_pos, cam_up_vector, target_pos)
    self.renderer = Renderer(self.workspace)
    self.pers_sensor = Sensor(cam_pos, cam_up_vector, target_pos, self.obs_size_m, cam_pos[2] - 1, cam_pos[2])

  def resetPybulletWorkspace(self):
    self.renderer.clearPoints()
    super().resetPybulletWorkspace()
    self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
    self.simulate_pos = self.robot._getEndEffectorPosition()
    self.simulate_rot = transformations.euler_from_quaternion(self.robot._getEndEffectorRotation())

  def step(self, action):
    self.robot.step = 0
    p, x, y, z, rot = self._decodeAction(action)
    current_pos = self.robot._getEndEffectorPosition()
    current_rot = list(transformations.euler_from_quaternion(self.robot._getEndEffectorRotation()))
    if self.action_sequence.count('r') == 1:
      current_rot[0] = 0
      current_rot[1] = 0

    pos = np.array(current_pos) + np.array([x, y, z])
    rot = np.array(current_rot) + np.array(rot)
    rot_q = pb.getQuaternionFromEuler(rot)
    pos[0] = np.clip(pos[0], self.workspace[0, 0], self.workspace[0, 1])
    pos[1] = np.clip(pos[1], self.workspace[1, 0], self.workspace[1, 1])
    pos[2] = np.clip(pos[2], self.workspace[2, 0], self.workspace[2, 1])
    self.robot.moveTo(pos, rot_q, dynamic=True)
    self.robot.controlGripper(p)
    self.robot.adjustGripperCommand()
    self.setRobotHoldingObj()
    self.renderer.clearPoints()
    obs = self._getObservation(action)
    valid = self.isSimValid()
    if valid:
      done = self._checkTermination()
      reward = self._getReward()
    else:
      done = True
      reward = 0
    if not done:
      done = self.current_episode_steps >= self.max_steps
    self.current_episode_steps += 1

    self.simulate_pos = pos
    self.simulate_rot = rot
    return obs, reward, done

  def _getReward(self):
    if self._checkTermination():
      return 1
    else:
      return 0

  def _getObservation(self, action=None):
    ''''''
    if self.obs_type == 'pixel':
      self.heightmap = self._getHeightmap()
      heightmap = self.heightmap
      #heightmap += np.clip(npr.normal(scale=1e-2, size=heightmap.shape), 0, 100)
      # draw gripper if view is centered at the gripper
      if self.view_type.find('camera_center_xy') > -1:
        gripper_img = self.robot.getGripperImg(self.heightmap_size, self.workspace_size, self.obs_size_m)
        if self.view_type.find('height') > -1:
          gripper_pos = self.robot._getEndEffectorPosition()
          heightmap[gripper_img == 1] = gripper_pos[2]
        else:
          heightmap[gripper_img == 1] = 0
      # add channel dimension if view is depth only
      if self.view_type.find('rgb') == -1:
        heightmap = heightmap.reshape([1, self.heightmap_size, self.heightmap_size])
      return self._isHolding(), None, heightmap
    else:
      obs = self._getVecObservation()
      return self._isHolding(), None, obs

  def _getHeightmap(self, gripper_pos=None, gripper_rz=None):
    if gripper_pos is None:
      gripper_pos = self.robot._getEndEffectorPosition()
    if gripper_rz is None:
      gripper_rz = transformations.euler_from_quaternion(self.robot._getEndEffectorRotation())[2]
    if self.view_type == 'render_center':
      return self.renderer.getTopDownDepth(self.obs_size_m, self.heightmap_size, gripper_pos, 0)
    elif self.view_type == 'render_center_height':
      depth = self.renderer.getTopDownDepth(self.obs_size_m, self.heightmap_size, gripper_pos, 0)
      heightmap = gripper_pos[2] - depth
      return heightmap
    elif self.view_type == 'render_fix':
      return self.renderer.getTopDownHeightmap(self.heightmap_size)

    elif self.view_type == 'camera_center_xyzr':
      # xyz centered, alighed
      gripper_pos[2] += self.robot.gripper_z_offset
      target_pos = [gripper_pos[0], gripper_pos[1], 0]
      T = transformations.euler_matrix(0, 0, gripper_rz)
      cam_up_vector = T.dot(np.array([-1, 0, 0, 1]))[:3]
      self.sensor.setCamMatrix(gripper_pos, cam_up_vector, target_pos)
      heightmap = self.sensor.getHeightmap(self.heightmap_size)
      depth = -heightmap + gripper_pos[2]
      return depth
    elif self.view_type in ['camera_center_xyr', 'camera_center_xyr_height']:
      # xy centered, aligned
      target_pos = [gripper_pos[0], gripper_pos[1], 0]
      T = transformations.euler_matrix(0, 0, gripper_rz)
      cam_up_vector = T.dot(np.array([-1, 0, 0, 1]))[:3]
      cam_pos = [gripper_pos[0], gripper_pos[1], 0.29]
      self.sensor.setCamMatrix(cam_pos, cam_up_vector, target_pos)
      heightmap = self.sensor.getHeightmap(self.heightmap_size)
      if self.view_type == 'camera_center_xyr':
        depth = -heightmap + gripper_pos[2]
      else:
        depth = heightmap
      return depth
    elif self.view_type in ['camera_center_xyz', 'camera_center_xyz_height']:
      # xyz centered, gripper will be visible
      gripper_pos[2] += self.robot.gripper_z_offset
      target_pos = [gripper_pos[0], gripper_pos[1], 0]
      cam_up_vector = [-1, 0, 0]
      self.sensor.setCamMatrix(gripper_pos, cam_up_vector, target_pos)
      heightmap = self.sensor.getHeightmap(self.heightmap_size)
      if self.view_type == 'camera_center_xyz':
        depth = -heightmap + gripper_pos[2]
      else:
        depth = heightmap
      return depth
    elif self.view_type in ['pers_center_xyz']:
      # xyz centered, gripper will be visible
      gripper_pos[2] += self.robot.gripper_z_offset
      target_pos = [gripper_pos[0], gripper_pos[1], 0]
      cam_up_vector = [-1, 0, 0]
      self.pers_sensor.setCamMatrix(gripper_pos, cam_up_vector, target_pos)
      heightmap = self.pers_sensor.getHeightmap(self.heightmap_size)
      depth = -heightmap + gripper_pos[2]
      return depth
    elif self.view_type in ['camera_center_xy', 'camera_center_xy_height']:
      # xy centered
      target_pos = [gripper_pos[0], gripper_pos[1], 0]
      cam_up_vector = [-1, 0, 0]
      cam_pos = [gripper_pos[0], gripper_pos[1], 0.29]
      self.sensor.setCamMatrix(cam_pos, cam_up_vector, target_pos)
      heightmap = self.sensor.getHeightmap(self.heightmap_size)
      if self.view_type == 'camera_center_xy':
        depth = -heightmap + gripper_pos[2]
      else:
        depth = heightmap
      return depth
    elif self.view_type in ['camera_center_z', 'camera_center_z_height']:
      gripper_pos[2] += self.robot.gripper_z_offset
      cam_pos = [self.workspace[0].mean(), self.workspace[1].mean(), gripper_pos[2]]
      target_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 0]
      cam_up_vector = [-1, 0, 0]
      self.sensor.setCamMatrix(cam_pos, cam_up_vector, target_pos)
      heightmap = self.sensor.getHeightmap(self.heightmap_size)
      if self.view_type == 'camera_center_z':
        depth = -heightmap + gripper_pos[2]
      else:
        depth = heightmap
      return depth
    elif self.view_type in ['camera_fix', 'camera_fix_height']:
      heightmap = self.sensor.getHeightmap(self.heightmap_size)
      if self.view_type == 'camera_fix':
        depth = -heightmap + gripper_pos[2]
      else:
        depth = heightmap
      return depth
    elif self.view_type in ['camera_side', 'camera_side_rgbd', 'camera_side_height']:
      cam_pos = [1, self.workspace[1].mean(), 0.6]
      target_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 0]
      cam_up_vector = [-1, 0, 0]
      self.sensor = Sensor(cam_pos, cam_up_vector, target_pos, 0.7, 0.1, 3)
      self.sensor.fov = 40
      self.sensor.proj_matrix = pb.computeProjectionMatrixFOV(self.sensor.fov, 1, self.sensor.near, self.sensor.far)
      if self.view_type == 'camera_side':
        depth = self.sensor.getDepthImg(self.heightmap_size)
      elif self.view_type == 'camera_side_rgbd':
        rgb_img = self.sensor.getRGBImg(self.heightmap_size)
        depth_img = self.sensor.getDepthImg(self.heightmap_size).reshape(1, self.heightmap_size, self.heightmap_size)
        depth = np.concatenate([rgb_img, depth_img])
      else:
        depth = self.sensor.getHeightmap(self.heightmap_size)
      return depth
    else:
      raise NotImplementedError


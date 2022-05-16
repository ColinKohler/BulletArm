import numpy as np
from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils.constants import NoValidPositionException
from bulletarm.pybullet.equipments.tray import Tray
from scipy.ndimage.interpolation import rotate
import pybullet as pb

class RandomHouseholdPickingClutterEnv(BaseEnv):
  '''
  '''
  def __init__(self, config):
    # env specific parameters
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.6, 0.6]
    if 'num_objects' not in config:
      config['num_objects'] = 3
    if 'max_steps' not in config:
      config['max_steps'] = 10
    super(RandomHouseholdPickingClutterEnv, self).__init__(config)
    self.object_init_z = 0.1
    self.obj_grasped = 0
    self.tray = Tray()

  def initialize(self):
    super().initialize()
    self.tray.initialize(pos=[self.workspace[0].mean(), self.workspace[1].mean(), 0],
                         size=[self.workspace_size+0.015, self.workspace_size+0.015, 0.1])

  def _decodeAction(self, action):
    """
    decode input action base on self.action_sequence
    Args:
      action: action tensor

    Returns: motion_primative, x, y, z, rot

    """
    primative_idx, x_idx, y_idx, z_idx, rot_idx = map(lambda a: self.action_sequence.find(a), ['p', 'x', 'y', 'z', 'r'])
    motion_primative = action[primative_idx] if primative_idx != -1 else 0
    if self.action_sequence.count('r') <= 1:
      rz = action[rot_idx] if rot_idx != -1 else 0
      ry = 0
      rx = 0
    else:
      raise NotImplementedError
    x = action[x_idx]
    y = action[y_idx]
    z = action[z_idx] if z_idx != -1 else self.getPatch_z(24, x, y, rz)

    rot = (rx, ry, rz)

    return motion_primative, x, y, z, rot


  def getPatch_z(self, patch_size, x, y, rz):
        """
        get the image patch in heightmap, centered at center_pixel, rotated by rz
        :param obs: BxCxHxW
        :param center_pixel: Bx2
        :param rz: B
        :return: image patch
        """
        img_size = self.heightmap_size
        x_pixel, y_pixel = self._getPixelsFromPos(x, y)
        center_pixel = np.array([y_pixel, x_pixel])
        transition = center_pixel - np.array([self.heightmap_size / 2, self.heightmap_size / 2])
        R = np.asarray([[np.cos(rz), np.sin(rz)],
                        [-np.sin(rz), np.cos(rz)]])
        rotated_transition = R.dot(transition) + np.array([self.heightmap_size / 2, self.heightmap_size / 2])

        rotated_heightmap = rotate(self.heightmap, angle=rz * 180 / np.pi, reshape=False)
        # patch = rotated_heightmap[int(rotated_transition[0] - patch_size / 2):
        #                           int(rotated_transition[0] + patch_size / 2),
        #                           int(rotated_transition[1] - patch_size / 2):
        #                           int(rotated_transition[1] + patch_size / 2)]
        patch = rotated_heightmap[int(rotated_transition[0] - 6):
                                  int(rotated_transition[0] + 6),
                                  int(rotated_transition[1] - patch_size / 2):
                                  int(rotated_transition[1] + patch_size / 2)]
        z = (np.min(patch) + np.max(patch)) / 2
        gripper_depth = 0.015
        gripper_reach = 0.02
        safe_z_pos = max(z, np.max(patch) - gripper_depth, np.min(patch) + gripper_reach)
        return safe_z_pos


  def step(self, action):
    pre_obj_grasped = self.obj_grasped
    self.takeAction(action)
    self.wait(200)
    # remove obj that above a threshold hight
    # for obj in self.objects:
    #   if obj.getPosition()[2] > self.pick_pre_offset:
    #     # self.objects.remove(obj)
    #     # pb.removeBody(obj.object_id)
    #     self._removeObject(obj)

    # for obj in self.objects:
    #   if not self._isObjectWithinWorkspace(obj):
    #     self._removeObject(obj)

    obs = self._getObservation(action)
    done = self._checkTermination()
    if self.reward_type == 'dense':
      reward = 1.0 if self.obj_grasped > pre_obj_grasped else 0.0
    else:
      reward = 1.0 if done else 0.0

    if not done:
      done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
    self.current_episode_steps += 1

    return obs, reward, done

  def reset(self):
    ''''''
    while True:
      self.resetPybulletWorkspace()
      try:
        # self.trayUid = pb.loadURDF(os.path.join(pybullet_data.getDataPath(), "tray/tray.urdf"),
        #                            self.workspace[0].mean(), self.workspace[1].mean(), 0,
        #                            0.000000, 0.000000, 1.000000, 0.000000)
        for i in range(self.num_obj):
          x = (np.random.rand() - 0.5) * 0.1
          x += self.workspace[0].mean()
          y = (np.random.rand() - 0.5) * 0.1
          y += self.workspace[1].mean()
          randpos = [x, y, 0.20]
          obj = self._generateShapes(constants.RANDOM_HOUSEHOLD, 1, random_orientation=self.random_orientation,
                                     pos=[randpos], padding=self.min_boarder_padding,
                                     min_distance=self.min_object_distance)
          pb.changeDynamics(obj[0].object_id, -1, lateralFriction=0.6)
      except NoValidPositionException:
        continue
      else:
        break
    self.obj_grasped = 0
    self.num_in_tray_obj = self.num_obj
    return self._getObservation()

  def isObjInBox(self, obj_pos, tray_pos, tray_size):
    tray_range = self.tray_range(tray_pos, tray_size)
    return tray_range[0][0] < obj_pos[0] < tray_range[0][1] and tray_range[1][0] < obj_pos[1] < tray_range[1][1]

  @staticmethod
  def tray_range(tray_pos, tray_size):
    return np.array([[tray_pos[0] - tray_size[0] / 2, tray_pos[0] + tray_size[0] / 2],
                     [tray_pos[1] - tray_size[1] / 2, tray_pos[1] + tray_size[1] / 2]])

  def InBoxObj(self, tray_pos, tray_size):
    obj_list = []
    for obj in self.objects:
      if self.isObjInBox(obj.getPosition(), tray_pos, tray_size):
        obj_list.append(obj)
    return obj_list

  def _checkTermination(self):
    ''''''
    for obj in self.objects:
      if self._isObjectHeld(obj):
        self.obj_grasped += 1
        self._removeObject(obj)
        if self.obj_grasped == self.num_obj:
          return True
        return False
    return False

  # def _checkTermination(self):
  #   ''''''
  #   self.num_in_tray_obj = len(self.InBoxObj([self.workspace[0].mean(), self.workspace[1].mean(), 0],
  #                                      [self.workspace_size+0.02, self.workspace_size+0.02, 0.02]))
  #   return self.num_in_tray_obj == 0

  def _getObservation(self, action=None):
    state, in_hand, obs = super(RandomHouseholdPickingClutterEnv, self)._getObservation()
    return 0, np.zeros_like(in_hand), obs

def createRandomHouseholdPickingClutterEnv(config):
  return RandomHouseholdPickingClutterEnv(config)

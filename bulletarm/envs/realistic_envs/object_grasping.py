import os
import math
import glob
import numpy as np
import bulletarm
from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils.constants import NoValidPositionException
from bulletarm.pybullet.equipments.tray import Tray
from scipy.ndimage.interpolation import rotate
import pybullet as pb

class ObjectGrasping(BaseEnv):
    '''Open loop object grasping task.

    The robot needs to pick up and object in a cluttered scene containing N random objects.
    The number of blocks N is set by the config.

    Args:
        config (dict): Intialization arguments for the env
    '''
    def __init__(self, config):
        # env specific parameters
        if 'object_scale_range' not in config:
            config['object_scale_range'] = [1, 1]
        if 'num_objects' not in config:
            config['num_objects'] = 15
        if 'max_steps' not in config:
            config['max_steps'] = 30
        config['adjust_gripper_after_lift'] = True
        config['min_object_distance'] = 0.
        config['min_boarder_padding'] = 0.15
        super(ObjectGrasping, self).__init__(config)
        self.object_init_z = 0.1
        self.obj_grasped = 0
        self.tray = Tray()
        self.exhibit_env_obj = False
        # self.exhibit_env_obj = True
        self.bin_size = self.workspace_size - 0.1
        self.gripper_depth = 0.04
        self.gripper_clearance = 0.01
        self.initialized = False

    def initialize(self):
        super().initialize()
        self.tray.initialize(pos=[self.workspace[0].mean(), self.workspace[1].mean(), 0],
                             size=[self.bin_size + 0.03, self.bin_size + 0.03, 0.1])
        self.initialized = True

    def _decodeAction(self, action):
        """
    decode input action base on self.action_sequence
    Args:
      action: action tensor

    Returns: motion_primative, x, y, z, rot

    """
        primative_idx, x_idx, y_idx, z_idx, rot_idx = map(lambda a: self.action_sequence.find(a),
                                                          ['p', 'x', 'y', 'z', 'r'])
        motion_primative = action[primative_idx] if primative_idx != -1 else 0
        if self.action_sequence.count('r') <= 1:
            rz = action[rot_idx] if rot_idx != -1 else 0
            ry = 0
            rx = 0
        else:
            raise NotImplementedError
        x = action[x_idx]
        y = action[y_idx]
        # x += (self.workspace[0, 0] + self.workspace[0, 1]) / 2
        # y += (self.workspace[1, 0] + self.workspace[1, 1]) / 2
        if z_idx != -1:
            z = action[z_idx]
        else:
            z = self.getPatch_z(x, y, rz)

        rot = (rx, ry, rz)

        return motion_primative, x, y, z, rot

    def _getPixelsFromPos(self, x, y):
        row_pixel, col_pixel = super()._getPixelsFromPos(x, y)
        row_pixel = min(row_pixel, self.heightmap_size - self.in_hand_size / 2 - 1)
        row_pixel = max(row_pixel, self.in_hand_size / 2)
        col_pixel = min(col_pixel, self.heightmap_size - self.in_hand_size / 2 - 1)
        col_pixel = max(col_pixel, self.in_hand_size / 2)
        return row_pixel, col_pixel

    def getPatch_z(self, x, y, rz, z=None):
        """
        get the image patch in heightmap, centered at center_pixel, rotated by rz
        :param obs:
        :param center_pixel
        :param rz:
        :return: safe z
        """
        row_pixel, col_pixel = self._getPixelsFromPos(x, y)
        # local_region is as large as ih_img
        local_region = self.heightmap[int(row_pixel - self.in_hand_size / 2): int(row_pixel + self.in_hand_size / 2),
                       int(col_pixel - self.in_hand_size / 2): int(col_pixel + self.in_hand_size / 2)]
        local_region = rotate(local_region, angle=-rz * 180 / np.pi, reshape=False)
        patch = local_region[int(self.in_hand_size / 2 - 16):int(self.in_hand_size / 2 + 16),
                int(self.in_hand_size / 2 - 4):int(self.in_hand_size / 2 + 4)]
        if z is None:
            edge = patch.copy()
            edge[5:-5] = 0
            # safe_z_pos = max(np.mean(patch.flatten()[(-patch).flatten().argsort()[2:12]]) - self.gripper_depth,
            #                  np.mean(edge.flatten()[(-edge).flatten().argsort()[:6]]) - 0.005)
            safe_z_pos = np.mean(patch.flatten()[(-patch).flatten().argsort()[2:12]]) - self.gripper_depth
            safe_z_pos += self.workspace[2, 0]
        else:
            safe_z_pos = np.mean(patch.flatten()[(-patch).flatten().argsort()[2:12]]) + z

        # use clearance to prevent gripper colliding with ground
        safe_z_pos = max(safe_z_pos, self.workspace[2, 0] + self.gripper_clearance)
        safe_z_pos = min(safe_z_pos, self.workspace[2, 1])
        assert self.workspace[2][0] <= safe_z_pos <= self.workspace[2][1]

        return safe_z_pos

    def _checkPerfectGrasp(self, x, y, z, rot, objects):
        return True

    def step(self, action):
        pre_obj_grasped = self.obj_grasped
        self.takeAction(action)
        self.wait(100)
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
        reward = 1.0 if self.obj_grasped > pre_obj_grasped else 0.0

        self.current_episode_steps += 1

        return obs, reward, True

    def isSimValid(self):
        for obj in self.objects:
            p = obj.getPosition()
            if not self.check_random_obj_valid and self.object_types[obj] == constants.RANDOM:
                continue
            if obj.getPosition()[2] >= 0.35 or self._isObjectHeld(obj):
                continue
            if self.workspace_check == 'point':
                if not self._isPointInWorkspace(p):
                    return False
            else:
                if not self._isObjectWithinWorkspace(obj):
                    return False
            if self.pos_candidate is not None:
                if np.abs(self.pos_candidate[0] - p[0]).min() > 0.02 or np.abs(
                    self.pos_candidate[1] - p[1]).min() > 0.02:
                    return False
        return True

    def reset(self):
        ''''''
        if not self.initialized or self.obj_grasped == self.num_obj or self.current_episode_steps > self.max_steps or not self.isSimValid():
            while True:
                self.resetPybulletWorkspace()
                try:
                    if not self.exhibit_env_obj:
                        for i in range(self.num_obj):
                            x = (np.random.rand() - 0.5) * 0.1
                            x += self.workspace[0].mean()
                            y = (np.random.rand() - 0.5) * 0.1
                            y += self.workspace[1].mean()
                            randpos = [x, y, 0.40]
                            # obj = self._generateShapes(constants.RANDOM_HOUSEHOLD, 1, random_orientation=self.random_orientation,
                            #                            pos=[randpos], padding=self.min_boarder_padding,
                            #                            min_distance=self.min_object_distance, model_id=-1)
                            # obj = self._generateShapes(constants.RANDOM_HOUSEHOLD200, 1,
                            #                            random_orientation=self.random_orientation,
                            #                            pos=[randpos], padding=self.min_boarder_padding,
                            #                            min_distance=self.min_object_distance, model_id=-1)

                            obj = self._generateShapes(constants.GRASP_NET_OBJ, 1,
                                                       random_orientation=self.random_orientation,
                                                       pos=[randpos], padding=self.min_boarder_padding,
                                                       min_distance=self.min_object_distance, model_id=-1)
                            pb.changeDynamics(obj[0].object_id, -1, lateralFriction=0.6)
                            self.wait(10)
                    # elif True:
                    # #create ducks
                    #     for i in range(15):
                    #         x = (np.random.rand() - 0.5) * 0.1
                    #         x += self.workspace[0].mean()
                    #         y = (np.random.rand() - 0.5) * 0.1
                    #         y += self.workspace[1].mean()
                    #         randpos = [x, y, 0.20]
                    #         creat_duck(randpos)
                    #         self.wait(100)
                    elif self.exhibit_env_obj:  # exhibit all random objects in this environment
                        # root_dir = os.path.dirname(bulletarm.__file__)
                        # urdf_pattern = os.path.join(root_dir, constants.URDF_PATH, 'random_household_object_200/*/*/*.obj')
                        # found_object_directories = glob.glob(urdf_pattern)
                        # total_num_objects = len(found_object_directories)
                        root_dir = os.path.dirname(bulletarm.__file__)
                        urdf_pattern = os.path.join(root_dir, constants.URDF_PATH, 'GraspNet1B_object/0*/')
                        found_object_directories = glob.glob(urdf_pattern)
                        total_num_objects = len(found_object_directories)

                        display_size = 1.5
                        columns = math.floor(math.sqrt(total_num_objects))
                        distance = display_size / (columns - 1)

                        obj_centers = []
                        obj_scales = []

                        for i in range(total_num_objects):
                            x = (i // columns) * distance
                            x += self.workspace[0].mean() + 0.6
                            y = (i % columns) * distance
                            y += self.workspace[1].mean() - display_size / 2
                            display_pos = [x, y, 0.08]
                            obj = self._generateShapes(constants.GRASP_NET_OBJ, 1,
                                                       rot=[pb.getQuaternionFromEuler([0., 0., -np.pi / 4])],
                                                       pos=[display_pos], padding=self.min_boarder_padding,
                                                       min_distance=self.min_object_distance, model_id=i)
                            # obj = self._generateShapes(constants.RANDOM_HOUSEHOLD200, 1,
                            #                            rot=[pb.getQuaternionFromEuler([0., 0., -np.pi / 4])],
                            #                            pos=[display_pos], padding=self.min_boarder_padding,
                            #                            min_distance=self.min_object_distance, model_id=i)
                        #     obj_centers.append(obj[0].center)
                        #     obj_scales.append(obj[0].real_scale)
                        #
                        # obj_centers = np.array(obj_centers)
                        # obj_scales = np.array(obj_scales)
                        print('Number of all objects: ', total_num_objects)
                        self.wait(10000)
                except NoValidPositionException:
                    continue
                else:
                    break
            self.wait(200)
            self.obj_grasped = 0
            # self.num_in_tray_obj = self.num_obj
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
            # if self._isObjectHeld(obj):
            #   self.obj_grasped += 1
            #   self._removeObject(obj)
            #   if self.obj_grasped == self.num_obj:
            #     return True
            #   return False
            if obj.getPosition()[2] >= 0.35 or self._isObjectHeld(obj):
                # ZXP getPos z > threshold is more robust than _isObjectHeld()
                self.obj_grasped += 1
                self._removeObject(obj)
                if self.obj_grasped == self.num_obj or len(self.objects) == 0:
                    return True
                return False
        return False

    def _getObservation(self, action=None):
        state, in_hand, obs = super(ObjectGrasping, self)._getObservation()
        return 0, np.zeros_like(in_hand), obs


def createObjectGrasping(config):
    return ObjectGrasping(config)

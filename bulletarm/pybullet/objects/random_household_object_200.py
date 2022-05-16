import sys

sys.path.append('..')

import pybullet as pb
import numpy as np
import os
import glob

import bulletarm
from bulletarm.pybullet.objects.pybullet_object import PybulletObject
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.objects.random_household_object_200_info import *

root_dir = os.path.dirname(bulletarm.__file__)
obj_pattern = os.path.join(root_dir, constants.OBJECTS_PATH, 'random_household_object_200/3dnet/*/*.obj')
found_object_directories = sorted(glob.glob(obj_pattern))
total_num_objects = len(found_object_directories)


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def delete_big_files():
    """
  deleting all big files while keep the smallest .obj file
  :return:
  """

    base = os.path.join(root_dir, constants.URDF_PATH, 'random_household_object_200/3dnet/')
    obj_classes = filter(lambda x: (x[0] != '.'), get_immediate_subdirectories(base))
    for obj_class in sorted(obj_classes):
        processing_obj_pattern = os.path.join(base, obj_class, '*.obj')
        found_object_directories = glob.glob(processing_obj_pattern)
        obj_file_keep = sorted(found_object_directories, key=lambda x: os.path.getsize(x))[0]
        for root, dirs, files in os.walk(os.path.join(base, obj_class)):
            pass
        print(obj_file_keep)
        for file in files:
            file_path = os.path.join(root, file)
            if file_path != obj_file_keep:
                os.remove(file_path)


class RandomHouseHoldObject200(PybulletObject):
    def __init__(self, pos, rot, scale, index=-1):
        if index < 0:
            while True:
                index = np.random.choice(np.arange(total_num_objects), 1)[0]
                if obj_avg_sr[index] > 0.97:  # sr > 0.8:  143 objs; sr > 0.9: 134 objs;
                                              # sr > 0.95: 105 objs; sr > 0.99: 41 objs;
                                              # sr > 0.97: 76 objs
                                              # avg sr = 0.923; avg sr > 0.8 = 0.968
                                              # avg sr > 0.97 = 0.9938
                    break
        else:
            sorted_obj_sr_indx = np.argsort(obj_avg_sr)[::-1]
            index = sorted_obj_sr_indx[index]
        obj_filepath = found_object_directories[index]
        # object_id = pb.loadURDF(obj_filepath, basePosition=pos, baseOrientation=rot, globalScaling=scale)

        color = np.random.uniform(0.6, 1, (4,))
        color[-1] = 1

        real_scale = obj_scales[index] * scale
        center = [x * scale for x in obj_centers[index]]
        obj_visual = pb.createVisualShape(pb.GEOM_MESH,
                                          fileName=obj_filepath,
                                          meshScale=[real_scale, real_scale, real_scale],
                                          rgbaColor=color,
                                          visualFramePosition=center)
        obj_collision = pb.createCollisionShape(pb.GEOM_MESH,
                                                fileName=obj_filepath,
                                                meshScale=[real_scale, real_scale, real_scale],
                                                collisionFramePosition=center)
        self.center = center
        self.real_scale = real_scale

        object_id = pb.createMultiBody(baseMass=0.15,
                                       baseCollisionShapeIndex=obj_collision,
                                       baseVisualShapeIndex=obj_visual,
                                       basePosition=pos,
                                       baseOrientation=rot)

        pb.changeDynamics(object_id,
                          -1,
                          lateralFriction=1,
                          spinningFriction=0.005,
                          rollingFriction=0.005)

        super(RandomHouseHoldObject200, self).__init__(constants.RANDOM, object_id)

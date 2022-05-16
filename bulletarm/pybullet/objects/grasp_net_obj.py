import sys

sys.path.append('..')

import pybullet as pb
import numpy as np
import os
import glob

import bulletarm
from bulletarm.pybullet.objects.pybullet_object import PybulletObject
from bulletarm.pybullet.utils import constants

root_dir = os.path.dirname(bulletarm.__file__)
obj_pattern = os.path.join(root_dir, constants.OBJECTS_PATH, 'GraspNet1B_object/0*/')
found_object_directories = sorted(glob.glob(obj_pattern))
total_num_objects = len(found_object_directories)


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


class GraspNetObject(PybulletObject):
    def __init__(self, pos, rot, scale, index=-1):

        if index >= 0:
            obj_filepath = found_object_directories[index]
        else:
            index = np.random.choice(np.arange(total_num_objects), 1)[0]
            obj_filepath = found_object_directories[index]

        color = np.random.uniform(0.6, 1, (4,))
        color[-1] = 1
        self.center = [0, 0, 0]
        obj_edge_max = 0.15 * scale  # the maximum edge size of an obj before scaling
        obj_edge_min = 0.014 * scale  # the minimum edge size of an obj before scaling
        obj_volume_max = 0.0006 * (scale ** 3)  # the maximum volume of an obj before scaling
        obj_scale = scale

        while True:
            obj_visual = pb.createVisualShape(pb.GEOM_MESH,
                                              fileName=obj_filepath + 'convex.obj',
                                              rgbaColor=color,
                                              meshScale=[obj_scale, obj_scale, obj_scale])
            obj_collision = pb.createCollisionShape(pb.GEOM_MESH,
                                                    fileName=obj_filepath + 'convex.obj',
                                                    meshScale=[obj_scale, obj_scale, obj_scale])

            object_id = pb.createMultiBody(baseMass=0.15,
                                           baseCollisionShapeIndex=obj_collision,
                                           baseVisualShapeIndex=obj_visual,
                                           basePosition=pos,
                                           baseOrientation=rot)

            aabb = pb.getAABB(object_id)
            aabb = np.asarray(aabb)
            size = aabb[1] - aabb[0]

            if np.partition(size, -2)[-2] > obj_edge_max:
                obj_scale *= 0.8
                pb.removeBody(object_id)
            elif size[0] * size[1] * size[2] > obj_volume_max:
                obj_scale *= 0.85
                pb.removeBody(object_id)
            elif size.min() < obj_edge_min:
                obj_scale /= 0.95
                pb.removeBody(object_id)
            else:
                break

        pb.changeDynamics(object_id,
                          -1,
                          lateralFriction=1,
                          spinningFriction=0.005,
                          rollingFriction=0.005)

        super(GraspNetObject, self).__init__(constants.RANDOM, object_id)

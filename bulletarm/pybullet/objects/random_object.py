import sys
sys.path.append('..')

import pybullet as pb
import numpy as np
import os
import glob
import re

import bulletarm
from bulletarm.pybullet.objects.pybullet_object import PybulletObject
from bulletarm.pybullet.utils import constants

root_dir = os.path.dirname(bulletarm.__file__)
obj_pattern = os.path.join(root_dir, constants.OBJECTS_PATH, 'random_urdfs/*/*.obj')
found_object_directories = glob.glob(obj_pattern)
found_object_directories = list(filter(lambda x: re.search(r'(002|005|027|032|066|075|083'
                                                           r'|116|118|131|137|142|143|149|154|176|187'
                                                           r'|199|200)\.obj', x),
                                       found_object_directories))
total_num_objects = len(found_object_directories)

class RandomObject(PybulletObject):
  def __init__(self, pos, rot, scale, z_scale=1):
    self.z_scale = z_scale
    obj_filepath = found_object_directories[np.random.choice(np.arange(total_num_objects), 1)[0]]
    mesh_scale = [0.01 * scale, 0.01 * scale, 0.01 * scale * z_scale]
    visualShapeId = pb.createVisualShape(shapeType=pb.GEOM_MESH,
                                         fileName=obj_filepath,
                                         rgbaColor=[np.random.random(), np.random.random(), np.random.random(), 1],
                                         meshScale=mesh_scale)
    collisionShapeId = pb.createCollisionShape(shapeType=pb.GEOM_MESH,
                                               fileName=obj_filepath,
                                               meshScale=mesh_scale)
    # collisionShapeId = pb.createCollisionShape(shapeType=pb.GEOM_BOX,
    #                                            halfExtents=[0.024*scale, 0.024*scale, 0.024*scale])
    object_id = pb.createMultiBody(baseMass=0.1,
                                   baseInertialFramePosition=[0, 0, 0],
                                   baseCollisionShapeIndex=collisionShapeId,
                                   baseVisualShapeIndex=visualShapeId,
                                   basePosition=pos,
                                   baseOrientation=rot)
    # pb.changeDynamics(object_id, -1, mass=0.1, lateralFriction=1.0, spinningFriction=0.0, rollingFriction=0.0)
    super(RandomObject, self).__init__(constants.RANDOM, object_id)

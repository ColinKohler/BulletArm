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
obj_pattern = os.path.join(root_dir, constants.OBJECTS_PATH, 'random_block/*.obj')
found_object_directories = glob.glob(obj_pattern)
total_num_objects = len(found_object_directories)

base_rotation_map = {
  '0.obj': (0, 0, np.pi/2),
  '1.obj': (np.pi/2, 0, np.pi/2),
  '2.obj': (0, np.pi/2, 1.8),
  '3.obj': (0, 0, np.pi/2),
  '4.obj': (0, 0, np.pi/2),
  '6.obj': (0, np.pi/2, np.pi/2),
  '7.obj': (0, 0, np.pi/2),
}

class RandomBlock(PybulletObject):
  def __init__(self, pos, rot, scale):
    obj_filepath = found_object_directories[np.random.choice(np.arange(total_num_objects), 1)[0]]
    model = obj_filepath.split('/')[-1]
    shape_rotation = pb.getQuaternionFromEuler(base_rotation_map[model]) if model in base_rotation_map.keys() else (0, 0, 0, 1)
    mesh_scale = [scale, scale, scale]
    visualShapeId = pb.createVisualShape(shapeType=pb.GEOM_MESH,
                                         fileName=obj_filepath,
                                         rgbaColor=[np.random.random(), np.random.random(), np.random.random(), 1],
                                         meshScale=mesh_scale,
                                         visualFrameOrientation=shape_rotation)
    collisionShapeId = pb.createCollisionShape(shapeType=pb.GEOM_MESH,
                                               fileName=obj_filepath,
                                               meshScale=mesh_scale,
                                               collisionFrameOrientation=shape_rotation)
    object_id = pb.createMultiBody(baseMass=0.1,
                                   baseInertialFramePosition=[0, 0, 0],
                                   baseCollisionShapeIndex=collisionShapeId,
                                   baseVisualShapeIndex=visualShapeId,
                                   basePosition=pos,
                                   baseOrientation=rot)
    super(RandomBlock, self).__init__(constants.RANDOM, object_id)

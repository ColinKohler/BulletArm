import numpy as np
import numpy.random as npr

from bulletarm.pybullet.utils import constants

# TODO:
#  - Random actions need to take z height into account
#  - Adding noise to actions should work w/all action possiblities i.e. (x y z rx ry rz)
#  - Figure out how to deal with place primitives
#  - Figure out if we should force specific obj ids or take object types for waypoints
#  - How to determine if an action was successful and should move on to the next waypoint

class WaypointPlanner(object):
  ''' Base waypoint planner class.

  Iterates through a list of waypoints to provide a plan for tasks.

  Args:
    env:
    config:
  '''
  def __init__(self, env, config):
    self.env = env
    self.pos_noise = config['pos_noise'] if 'pos_noise' in config else None
    self.rot_noise = config['rot_noise'] if 'rot_noise' in config else None
    self.random_orientation = config['random_orientation'] if 'random_orientation' in config else True
    self.half_rotation = config['half_rotation'] if 'half_rotation' in config else False
    self.waypoint_index = 0

    npr.seed(env.seed)

  def getNextAction(self):
    '''
    Get the next action from the list of waypoints.
    '''
    waypoint = self.env.planner_waypoints[self.waypoint_index]

    self.waypoint_index += 1
    if self.waypoint_index >= len(self.env.planner_waypoints):
      self.waypoint_index = 0

    primitive, target, _ = waypoint
    if target == constants.RANDOM:
      return self.getRandomAction(primitive)

    obj_pose = self.getObjectPose(target)
    x, y, z, rx, ry, rz = obj_pose

    return self.encodeAction(primitive, x, y, z, rz)

  def encodeAction(self, primitive, x, y, z, r):
    '''
    Encode the action into the format required by the enviornment.

    Args:
      primitive ():
      x ():
      y ():
      z ():
      r ():

    Returns:
      () :
    '''
    x, y, r = self.addNoiseToAction(x, y, r)
    if self.half_rotation:
      if not hasattr(r, '__len__'):
        while r < 0:
          r += np.pi
        while r > np.pi:
          r -= np.pi
      else:
        rz, ry, rx = r
        while rz < 0:
          rz += np.pi
          rx = -rx
          ry = -ry
        while rz > np.pi:
          rz -= np.pi
          rx = -rx
          ry = -ry
        r = rz, ry, rx
    return self.env._encodeAction(primitive, x, y, z, r)

  def getRandomAction(self, primitive):
    '''
    Get a random action of the specified primitive motion.

    Args:
      primitive (int): Motion primitive for the random action

    Returns:
      () : The random action
    '''
    x = npr.uniform(self.env.workspace[0, 0], self.env.workspace[0, 1])
    y = npr.uniform(self.env.workspace[1, 0], self.env.workspace[1, 1])
    z = 0. # TODO: Fix this
    r = npr.uniform(0., np.pi)
    return self.encodeAction(primitive, x, y, z, r)

  def addNoiseToAction(self, x, y, r):
    '''
    Add random noise to the action

    Args:
      x ():
      y ():
      r ():

    Returns:
      () :
    '''
    if self.pos_noise:
      x = np.clip(
        x + npr.uniform(-self.pos_noise, self.pos_noise),
        self.env.workspace[0,0],
        self.env.workspace[0,1]
      )
      y = np.clip(
        y + npr.uniform(-self.pos_noise, self.pos_noise),
        self.env.workspace[1,0],
        self.env.workspace[1,1]
      )

    if self.rot_noise:
      r = np.clip(
        r + npr.uniform(-self.rot_noise, self.rot_noise),
        0,
        np.pi
      )

    return x, y, r

  def getObjectPose(self, obj):
    '''

    '''
    objects = self.getObjects()
    object_pose = self.env.getObjectPoses([obj])

    return object_pose[0]

  def getObjects(self, obj_type=None):
    '''
    Get a list of objects in the environment not being held. If an object type is indicated
    only objects of that type will be returned.

    Args:
      obj_type (int): Object type to return. Defaults to None.

    Returns:
      list[int]: List of object ids in the enviornment.
    '''
    if obj_type is not None:
      return list(
        filter(lambda x: self.env.object_types[x] == obj_type or
               not self.env._isObjectHeld(x) or
               not self.env.isObjectInWorkspace(x),
               self.env.objects)
      )
    else:
      return list(filter(lambda x: not self.env._isObjectHeld(x), self.env.objects))

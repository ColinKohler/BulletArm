'''
.. moduleauthor: Colin Kohler <github.com/ColinKohler>
'''

import pybullet as pb

class PybulletObject(object):
  '''
  Base obejct class.

  Args:
    object_type_id (int): The id for the object type being loaded. See cosntants.py.
    object_id (int): The object id.
  '''
  def __init__(self, object_type_id, object_id):
    self.object_type_id = object_type_id
    self.object_id = object_id

  def getXPosition(self):
    '''
    Get the current x position of the object.

    Returns:
      float: The x position of the object.
    '''
    return self.getPosition()[0]

  def getYPosition(self):
    '''
    Get the current y position of the object.

    Returns:
      float: The y position of the object.
    '''
    return self.getPosition()[1]

  def getXYPosition(self):
    '''
    Get the current xy position of the object.

    Returns:
      list[float]: The xy position of the object.
    '''
    return self.getPosition()[:2]

  def getZPosition(self):
    '''
    Get the current z position of the object.

    Returns:
      float: The z position of the object.
    '''
    return self.getPosition()[2]

  def getPosition(self):
    '''
    Get the current xyz position of the object.

    Returns:
      list[float]: The xyz position of the object.
    '''
    pos, _ = pb.getBasePositionAndOrientation(self.object_id)
    return list(pos)

  def getRotation(self):
    '''
    Get the current rpy orientation of the object.

    Returns:
      list[float](numpy.array): The rpy orientation of the object.
    '''
    _, rot = pb.getBasePositionAndOrientation(self.object_id)
    return list(rot)

  def getPose(self):
    '''
    Get the current pose of the object.

    Returns:
      (list[float], list[float]): (position, orientation)
    '''
    pos, rot = pb.getBasePositionAndOrientation(self.object_id)
    return list(pos), list(rot)

  def getGraspPosition(self):
    '''
    Get the canonical grasp position for the object.

    Returns:
      list[float]: The grasp position
    '''
    return self.getPosition()

  def getGraspRotation(self):
    '''
    Get the canonical grasp orientation for the object.

    Returns:
      list[float]: The grasp orientation
    '''
    return self.getRotation()

  def getGraspPose(self):
    '''
    Get the canonical grasp pose for the object.

    Returns:
      list[float]: The grasp pose
    '''
    return self.getPose()

  def setVelocity(self, linear_velocity, angular_velocity):
    '''
    Set the velocity for the object.

    Args:
      linear_velocity (numpy.array): Linear velocity for the object.
      angular_velocity (numpy.array): Angular velocity for the object.

    Returns:

    '''
    return pb.resetBaseVelocity(self.object_id, linear_velocity, angular_velocity)

  def getVelocity(self):
    '''
    Get the velocity for the object.

    Returns:
      (list[float], list[float]): (linear_velocity, angular velocity)
    '''
    return pb.getBaseVelocity(self.object_id)

  def getBoundingBox(self):
    '''
    Get the bounding box for the object

    Returns:
      list[float]: The AABB bounding box
    '''
    return list(pb.getAABB(self.object_id))

  def getContactPoints(self):
    '''
    Get the current points of contact on the obejct.

    Returns:
      list[float]: The points of contact
    '''
    return pb.getContactPoints(self.object_id)

  def isTouching(self, obj):
    '''
    Checks to see if two objects are touching.

    Args:
      obj (PybulletObject): Other object

    Returns:
      Bool: True if objects are touching, False otherwise.
    '''
    contact_points = self.getContactPoints()
    for p in contact_points:
      if p[2] == obj.object_id:
        return True
    return False

  def isTouchingId(self, obj_id):
    '''
    Checks to see if two objects are touching.

    Args:
      obj_id (int): ID of the other object

    Returns:
      Bool: True if objects are touching, False otherwise.
    '''
    contact_points = self.getContactPoints()
    for p in contact_points:
      if p[2] == obj_id:
        return True
    return False

  def resetPose(self, pos, rot):
    '''
    Set the object to the given pose.

    Args:
      pos (numpy.array): Position
      rot (numpy.array): Orientation
    '''
    pb.resetBasePositionAndOrientation(self.object_id, pos, rot)

  def __eq__(self, other):
    '''
    Class comparitor. Used to check equality between objects, i.e. (a == b)

    Args:
      other (PybulletObject): Other object to check for comparison.

    Returns:
      Bool: Are the two objects the same.
    '''
    if not isinstance(other, PybulletObject):
      return False
    return self.object_id == other.object_id and self.object_type_id == other.object_type_id

  def __hash__(self):
    '''
    Object hash is the object ID. Used for saving environment state.

    Returns:
      int: Obejct ID
    '''
    return self.object_id

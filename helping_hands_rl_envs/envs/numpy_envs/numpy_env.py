import os
import json
import pickle
from copy import deepcopy
import numpy as np
import numpy.random as npr

from helping_hands_rl_envs.envs.base_env import BaseEnv
from helping_hands_rl_envs.simulators.numpy import object_generation
from helping_hands_rl_envs.simulators import constants

class NumpyEnv(BaseEnv):
  def __init__(self, config):
    if 'pick_rot' not in config:
      config['pick_rot'] = True
    if 'place_rot' not in config:
      config['place_rot'] = False
    if 'scale' not in config:
      config['scale'] = 1.
    if 'pos_candidate' not in config:
      config['pos_candidate'] = None
    if 'in_hand_size' not in config:
      config['in_hand_size'] = 24
    if 'in_hand_mode' not in config:
      config['in_hand_mode'] = 'sub'

    seed = config['seed']
    workspace = config['workspace']
    max_steps = config['max_steps']
    heightmap_size = config['obs_size']
    render = config['render']
    action_sequence = config['action_sequence']
    pos_candidate = config['pos_candidate']
    pick_rot = config['pick_rot']
    place_rot = config['place_rot']
    scale = config['scale']
    in_hand_size = config['in_hand_size']
    in_hand_mode = config['in_hand_mode']
    super(NumpyEnv, self).__init__(seed, workspace, max_steps, heightmap_size, action_sequence, pos_candidate,
                                   in_hand_size, in_hand_mode)

    self.scale = scale
    self.render = render
    self.offset = self.scale*self.heightmap_size/20
    self.valid = True
    self.pick_rot = pick_rot
    self.place_rot = place_rot

    # TODO: These are kinda pointless but the pick/place planners currently use them...
    self.pick_offset = 0.0
    self.place_offset = 0.0

  def reset(self):
    ''''''
    self.held_object = None
    self.heightmap = np.zeros((self.heightmap_size, self.heightmap_size))
    self.current_episode_steps = 1
    self.objects = list()
    self.valid = True

    return self._getObservation()

  def saveState(self):
    self.state = {'held_object_idx': self.objects.index(self.held_object) if self.objects and self.held_object else None,
                  'heightmap': deepcopy(self.heightmap),
                  'current_episode_steps': deepcopy(self.current_episode_steps),
                  'objects': deepcopy(self.objects),
                  'valid': deepcopy(self.valid)}

  def restoreState(self):
    self.heightmap = self.state['heightmap']
    self.current_episode_steps = self.state['current_episode_steps']
    self.objects = self.state['objects']
    self.valid = self.state['valid']
    held_object_idx = self.state['held_object_idx']
    self.held_object = self.objects[held_object_idx] if held_object_idx is not None else None

  def saveEnvToFile(self, path):
    # np_file = os.path.join(path, 'env.np')
    pickle_file = os.path.join(path, 'env.pickle')
    # np.save(np_file, self.heightmap)
    state = {
      'heightmap': self.heightmap,
      'held_object_idx': self.objects.index(self.held_object) if self.objects and self.held_object else None,
      'current_episode_steps': deepcopy(self.current_episode_steps),
      'objects': deepcopy(self.objects),
      'valid': deepcopy(self.valid)}
    with open(pickle_file, 'wb') as f:
      pickle.dump(state, f)

  def loadEnvFromFile(self, path):
    # np_file = os.path.join(path, 'env.np')
    # json_file = os.path.join(path, 'env.json')
    pickle_file = os.path.join(path, 'env.pickle')
    with open(pickle_file, 'rb') as f:
      state = pickle.load(f)
    self.heightmap = state['heightmap']
    self.current_episode_steps = state['current_episode_steps']
    self.objects = state['objects']
    self.valid = state['valid']
    held_object_idx = state['held_object_idx']
    self.held_object = self.objects[held_object_idx] if held_object_idx is not None else None

  def takeAction(self, action):
    motion_primative, x, y, z, rot = self._decodeAction(action)

    if motion_primative == constants.PICK_PRIMATIVE:
      self.held_object = self._pick(x, y, z, rot)
    elif motion_primative == constants.PLACE_PRIMATIVE:
      if self.held_object:
        self._place(x, y, z, rot)
        self.held_object = None
    elif motion_primative == constants.PUSH_PRIMATIVE:
      pass
    else:
      raise ValueError('Bad motion primative supplied for action.')
    self._fixTopObjects()

  def wait(self, iteration):
    pass

  def isSimValid(self):
    return self.valid

  def _checkPickValid(self, x, y, z, rot, check_rot):
    if self._isHolding():
      return False

    height_sorted_objects = sorted(self.objects, key=lambda x: x.pos[-1], reverse=True)
    for obj in height_sorted_objects:
      if obj.isGraspValid([x,y,z], rot, check_rot):
        return True
    return False

  def _checkPlaceValid(self, x, y, z, rot, check_rot):
    padding = self.scale * self.heightmap_size / 10
    x = max(x, padding)
    x = min(x, self.heightmap_size - padding)
    y = max(y, padding)
    y = min(y, self.heightmap_size - padding)

    if self.held_object is None:
      return False
    for i, obj in enumerate(self.objects):
      if self.held_object is obj or not obj.on_top:
        continue
      if self.held_object.isStackValid([x, y, z], rot, obj, check_rot):
        return True
      else:
        distance = np.linalg.norm(np.array([x, y]) - (obj.pos[:-1]))
        min_distance = np.sqrt(2)/2 * (self.held_object.size + obj.size)
        if distance < min_distance:
          return False
    return False

  def _pick(self, x, y, z, rot):
    ''''''
    if self._isHolding():
      return self.held_object

    height_sorted_objects = sorted(self.objects, key=lambda x: x.pos[-1], reverse=True)
    for obj in height_sorted_objects:
      if obj.isGraspValid([x,y,z], rot, self.pick_rot):
        obj.rot -= rot
        if obj.rot < 0:
          obj.rot += np.pi
        obj.removeFromHeightmap(self.heightmap)
        return obj

    return None

  def _place(self, x, y, z, rot):
    ''''''
    padding = self.scale * self.heightmap_size / 10
    x = max(x, padding)
    x = min(x, self.heightmap_size - padding)
    y = max(y, padding)
    y = min(y, self.heightmap_size - padding)

    if self.held_object is None:
      return
    for i, obj in enumerate(self.objects):
      if self.held_object is obj or not obj.on_top:
        continue
      if self.held_object.isStackValid([x, y, z], rot, obj, self.place_rot):
        self.held_object.addToHeightmap(self.heightmap, [x, y, z], rot)
        self._fixTopObjects()
        return
      else:
        distance = np.linalg.norm(np.array([x, y]) - (obj.pos[:-1]))
        min_distance = np.sqrt(2)/2 * (self.held_object.size + obj.size)
        if distance < min_distance:
          self.valid = False
          return
    self.held_object.addToHeightmap(self.heightmap, [x, y, z], rot)

  def _fixTopObjects(self):
    for obj in self.objects:
      if self.heightmap[obj.mask].mean() > obj.pos[-1]:
        obj.on_top = False
      else:
        obj.on_top = True

  def _getNumTopBlock(self, objects=None):
    if objects is None:
      objects = self.objects
    count = 0
    for obj in objects:
      if self.held_object == obj or obj.on_top:
        count += 1
    return count

  def _checkStack(self):
    return self._getNumTopBlock() == 1

  def _getNumTopCylinder(self):
    count = 0
    for obj in self._getCylinders():
      if self.held_object == obj:
        return -1
      if obj.on_top:
        count += 1
    return count

  def _getObservation(self, action=None):
    ''''''
    old_heightmap = self.heightmap.copy()
    if action is None or self._isHolding() == False:
      in_hand_img = np.zeros((self.in_hand_size, self.in_hand_size, 1))
    else:
      motion_primative, x, y, z, rot = self._decodeAction(action)
      in_hand_img = self.getInHandImage(old_heightmap, x, y, z, rot, self.heightmap)

    return self._isHolding(), in_hand_img, self.heightmap.reshape([self.heightmap_size, self.heightmap_size, 1])

  def _getValidPositions(self, padding, min_distance, existing_positions, num_shapes):
    while True:
      existing_positions_copy = deepcopy(existing_positions)
      valid_positions = []
      for i in range(num_shapes):
        # Generate random drop config
        x_extents = self.workspace[0][1] - self.workspace[0][0]
        y_extents = self.workspace[1][1] - self.workspace[1][0]

        is_position_valid = False
        for j in range(1000):
          if is_position_valid:
            break
          position = [int((x_extents - padding) * npr.random_sample() + self.workspace[0][0] + padding / 2),
                      int((y_extents - padding) * npr.random_sample() + self.workspace[1][0] + padding / 2)]
          if self.pos_candidate is not None:
            position[0] = self.pos_candidate[0][np.abs(self.pos_candidate[0] - position[0]).argmin()]
            position[1] = self.pos_candidate[1][np.abs(self.pos_candidate[1] - position[1]).argmin()]

          if existing_positions_copy:
            # is_position_valid = np.all(
            #   np.sum(np.abs(np.array(positions)[:, :2] - np.array(position)[:2]), axis=1) > min_distance)
            distances = np.array(list(map(lambda p: np.linalg.norm(np.array(p) - position), existing_positions_copy)))
            is_position_valid = np.all(distances > min_distance)
          else:
            is_position_valid = True
        if is_position_valid:
          existing_positions_copy.append(position)
          valid_positions.append(position)
        else:
          break
      if len(valid_positions) == num_shapes:
        return valid_positions

  def _generateShapes(self, object_type, num_objects, min_distance=None, padding=None, random_orientation=False):
    ''''''
    if min_distance is None:
      min_distance = 2 * self.scale*self.heightmap_size/7
    if padding is None:
      padding = self.scale*self.heightmap_size/5
    objects = list()
    positions = deepcopy(list(map(lambda o: o.pos[:-1], self.objects)))

    valid_positions = self._getValidPositions(padding, min_distance, positions, num_objects)
    for position in valid_positions:
      if random_orientation:
        rotation = np.pi * np.random.random_sample()
      else:
        rotation = 0.0
      size = npr.randint(self.scale * self.heightmap_size / 10, self.scale * self.heightmap_size / 7)
      position.append(int(size / 2))

      if object_type is constants.CUBE:
        obj, self.heightmap = object_generation.generateCube(self.heightmap, position, rotation, size)
      elif object_type is constants.CYLINDER:
        obj, self.heightmap = object_generation.generateCylinder(self.heightmap, position, rotation, size)
      else:
        raise NotImplementedError
      objects.append(obj)
    self.objects.extend(objects)
    return objects

  def _removeObject(self, obj):
    if obj == self.held_object:
      self.held_object = None
    self.objects.remove(obj)

  def _isHolding(self):
    return not (self.held_object is None)

  def _isObjectHeld(self, obj):
    return self.held_object == obj

  def _getObjectPosition(self, obj):
    ''''''
    return obj.pos

  def _isObjOnTop(self, obj, objects=None):
    if not objects:
      objects = self.objects
    obj_position = obj.getPosition()
    for o in objects:
      if self._isObjectHeld(o) or o is obj:
        continue
      block_position = o.getPosition()
      # TODO: This atol could be bad, the direct port from pybullet didnt work so I did this instead
      if np.allclose(block_position[:-1], obj_position[:-1],atol=0.01) and \
         block_position[-1] > obj_position[-1]:
        return False
    return True

  # TODO: Move this to base clase
  def getObjects(self):
    objs = list()
    for obj in self.objects:
      if self._isObjectHeld(obj):
        continue
      objs.append(obj)
    return np.array(objs)

  def getObjectPoses(self, objects=None):
    if objects is None: objects = self.objects

    obj_poses = list()
    for obj in objects:
      if self._isObjectHeld(obj):
        continue
      pos, rot = obj.getPose()
      obj_poses.append(pos + rot)
    return np.array(obj_poses)

  def _getBlocks(self):
    return list(filter(lambda o: type(o) is object_generation.Cube, self.objects))

  def _getCylinders(self):
    return list(filter(lambda o: type(o) is object_generation.Cylinder, self.objects))

  def planBlockPicking(self):
    # pick
    if self.held_object is None:
      height_sorted_objects = sorted(self.objects, key=lambda x: x.pos[-1])
      for obj in height_sorted_objects:
        if not obj.on_top:
          continue
        return self._encodeAction(self.PICK_PRIMATIVE, obj.pos[0], obj.pos[1], obj.pos[2] - 2, obj.rot)

  def planBlockStacking(self):
    # pick
    if self.held_object is None:
      height_sorted_objects = sorted(self.objects, key=lambda x: x.pos[-1])
      for obj in height_sorted_objects:
        if not obj.on_top:
          continue
        return self._encodeAction(self.PICK_PRIMATIVE, obj.pos[0], obj.pos[1], obj.pos[2]-2, obj.rot)

    # place
    else:
      reverse_height_sorted_objects = sorted(self.objects, key=lambda x: x.pos[-1], reverse=True)
      for obj in reverse_height_sorted_objects:
        if obj is self.held_object:
          continue
        rot = obj.rot - self.held_object.rot
        while rot < 0:
          rot += np.pi
        while rot > np.pi:
          rot -= np.pi
        return self._encodeAction(self.PLACE_PRIMATIVE, obj.pos[0], obj.pos[1], obj.pos[2]+2, rot)

  def planBlockStackingWithX(self, primitive, x, y):
    # pick
    if primitive == self.PICK_PRIMATIVE:
      sorted_objects = sorted(self.objects, key=lambda o: np.linalg.norm(np.array(o.pos[:2]) - np.array([x, y])))
      for obj in sorted_objects:
        if not obj.on_top:
          continue
        return self._encodeAction(self.PICK_PRIMATIVE, x, y, obj.pos[2]-2, obj.rot)

    # place
    else:
      sorted_objects = sorted(self.objects, key=lambda o: np.linalg.norm(np.array(o.pos[:2]) - np.array([x, y])))
      for obj in sorted_objects:
        if obj is self.held_object or not obj.on_top:
          continue
        rot = obj.rot - self.held_object.rot
        while rot < 0:
          rot += np.pi
        while rot > np.pi:
          rot -= np.pi
        return self._encodeAction(self.PLACE_PRIMATIVE, x, y, obj.pos[2]+2, rot)

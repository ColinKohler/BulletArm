from copy import deepcopy
import numpy as np
import numpy.random as npr

from helping_hands_rl_envs.envs.base_env import BaseEnv
from helping_hands_rl_envs.numpy_toolkit import object_generation

class NumpyEnv(BaseEnv):
  def __init__(self, seed, workspace, max_steps=10, heightmap_size=250, render=False, action_sequence='pxyr'):
    super(NumpyEnv, self).__init__(seed, workspace, max_steps, heightmap_size, action_sequence)

    self.render = render
    self.offset = self.heightmap_size/20
    self.valid = True

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

  def step(self, action):
    ''''''
    motion_primative, x, y, z, rot = self._getSpecificAction(action)

    if motion_primative == self.PICK_PRIMATIVE:
      self.held_object = self._pick(x, y, z, rot)
    elif motion_primative == self.PLACE_PRIMATIVE:
      if self.held_object:
        self._place(x, y, z, rot)
        self.held_object = None
    elif motion_primative == self.PUSH_PRIMATIVE:
      pass
    else:
      raise ValueError('Bad motion primative supplied for action.')

    # Check for termination and get reward
    obs = self._getObservation()
    done = self._checkTermination()
    reward = 1.0 if done else 0.0

    # Check to see if we are at the max step
    if not done:
      done = self.current_episode_steps >= self.max_steps or not self.valid
    self.current_episode_steps += 1

    return obs, reward, done

  def _pick(self, x, y, z, rot):
    ''''''
    if self._isHolding():
      return self.held_object

    height_sorted_objects = sorted(self.objects, key=lambda x: x.pos[-1], reverse=True)
    for obj in height_sorted_objects:
      if obj.isGraspValid([x,y,z], rot):
        obj.removeFromHeightmap(self.heightmap)
        return obj

    return None

  def _place(self, x, y, z, rot):
    ''''''
    padding = self.heightmap_size / 10
    x = max(x, padding)
    x = min(x, self.heightmap_size - padding)
    y = max(y, padding)
    y = min(y, self.heightmap_size - padding)

    if self.held_object is None:
      return
    for i, obj in enumerate(self.objects):
      if self.held_object is obj or not obj.on_top:
        continue
      if self.held_object.isStackValid([x, y, z], rot, obj):
        self.held_object.addToHeightmap(self.heightmap, [x, y, z], rot)
        self._fixTopBlocks()
        return
      else:
        distance = np.linalg.norm(np.array([x, y]) - (obj.pos[:-1]))
        min_distance = np.sqrt(2)/2 * (self.held_object.size + obj.size)
        if distance < min_distance:
          self.valid = False
          return
    self.held_object.addToHeightmap(self.heightmap, [x, y, z], rot)
    self._fixTopBlocks()

  def _fixTopBlocks(self):
    for obj in self.objects:
      if self.heightmap[obj.mask].mean() > obj.pos[-1]:
        obj.on_top = False
      else:
        obj.on_top = True

  def _getNumTopBlock(self):
    count = 0
    for obj in self.objects:
      if obj.on_top:
        count += 1
    return count

  def _getObservation(self):
    ''''''
    return self._isHolding(), self.heightmap.reshape([self.heightmap_size, self.heightmap_size, 1])

  def _generateShapes(self, object_type, num_objects, min_distance=None, padding=None, random_orientation=False):
    ''''''
    if min_distance is None:
      min_distance = 2 * self.heightmap_size/7
    if padding is None:
      padding = self.heightmap_size/5
    self.objects = list()
    positions = list()
    for i in range(num_objects):
      # Generate random drop config
      x_extents = self.workspace[0][1] - self.workspace[0][0]
      y_extents = self.workspace[1][1] - self.workspace[1][0]

      is_position_valid = False
      while not is_position_valid:
        position = [int((x_extents - padding) * npr.random_sample() + self.workspace[0][0] + padding / 2),
                    int((y_extents - padding) * npr.random_sample() + self.workspace[1][0] + padding / 2),
                    0]
        if positions:
          is_position_valid = np.all(np.sum(np.abs(np.array(positions) - np.array(position)), axis=1) > min_distance)
        else:
          is_position_valid = True

      positions.append(position)
      if random_orientation:
        rotation = np.pi*np.random.random_sample()
      else:
        rotation = 0.0
      size = npr.randint(self.heightmap_size/10, self.heightmap_size/7)
      position[2] = int(size / 2)

      obj, self.heightmap = object_generation.generateCube(self.heightmap, position, rotation, size)
      self.objects.append(obj)

    return self.objects

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

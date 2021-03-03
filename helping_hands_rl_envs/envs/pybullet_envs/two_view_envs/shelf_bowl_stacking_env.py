import os
import pybullet as pb
import copy
import numpy as np
import numpy.random as npr

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.equipments.shelf import Shelf
from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.utils import transformations
from helping_hands_rl_envs.simulators.constants import NoValidPositionException
from helping_hands_rl_envs.planners.shelf_bowl_stacking_planner import ShelfBowlStackingPlanner

class ShelfBowlStackingEnv(PyBulletEnv):
  def __init__(self, config):
    super().__init__(config)
    self.place_offset = 0.04
    self.shelf = Shelf()
    self.object_init_space = np.asarray([[0.3, 0.7],
                                         [-0.4, 0],
                                         [0, 0.40]])
  
  def initialize(self):
    super().initialize()
    self.shelf.initialize(pos=[0.6, 0.3, 0])

    pass

  def reset(self):
    ''''''
    while True:
      self.resetPybulletEnv()
      try:
        self._generateShapes(constants.BOWL, self.num_obj, random_orientation=self.random_orientation)
      except NoValidPositionException as e:
        continue
      else:
        break
    return self._getObservation()

  def anyObjectOnTarget1(self):
    for obj in self.objects:
      if self.shelf.isObjectOnTarget1(obj):
        return True
    return False

  def _checkTermination(self):
    return self._checkStack() and self.anyObjectOnTarget1()
    
  def _getValidPositions(self, border_padding, min_distance, existing_positions, num_shapes, sample_range=None):
    existing_positions_copy = copy.deepcopy(existing_positions)
    sample_range = copy.deepcopy(sample_range)
    valid_positions = list()
    for i in range(num_shapes):
      # Generate random drop config
      x_extents = self.object_init_space[0][1] - self.object_init_space[0][0]
      y_extents = self.object_init_space[1][1] - self.object_init_space[1][0]

      is_position_valid = False
      for j in range(100):
        if is_position_valid:
          break
        if sample_range:
          sample_range[0][0] = max(sample_range[0][0], self.object_init_space[0][0]+border_padding/2)
          sample_range[0][1] = min(sample_range[0][1], self.object_init_space[0][1]-border_padding/2)
          sample_range[1][0] = max(sample_range[1][0], self.object_init_space[1][0]+border_padding/2)
          sample_range[1][1] = min(sample_range[1][1], self.object_init_space[1][1]-border_padding/2)
          position = [(sample_range[0][1] - sample_range[0][0]) * npr.random_sample() + sample_range[0][0],
                      (sample_range[1][1] - sample_range[1][0]) * npr.random_sample() + sample_range[1][0]]
        else:
          position = [(x_extents - border_padding) * npr.random_sample() + self.object_init_space[0][0] + border_padding / 2,
                      (y_extents - border_padding) * npr.random_sample() + self.object_init_space[1][0] +  border_padding / 2]

        if self.pos_candidate is not None:
          position[0] = self.pos_candidate[0][np.abs(self.pos_candidate[0] - position[0]).argmin()]
          position[1] = self.pos_candidate[1][np.abs(self.pos_candidate[1] - position[1]).argmin()]
          if not (self.object_init_space[0][0]+border_padding/2 < position[0] < self.object_init_space[0][1]-border_padding/2 and
                  self.object_init_space[1][0]+border_padding/2 < position[1] < self.object_init_space[1][1]-border_padding/2):
            continue

        if existing_positions_copy:
          distances = np.array(list(map(lambda p: np.linalg.norm(np.array(p)-position), existing_positions_copy)))
          is_position_valid = np.all(distances > min_distance)
          # is_position_valid = np.all(np.sum(np.abs(np.array(positions) - np.array(position[:-1])), axis=1) > min_distance)
        else:
          is_position_valid = True
      if is_position_valid:
        existing_positions_copy.append(position)
        valid_positions.append(position)
      else:
        break
    if len(valid_positions) == num_shapes:
      return valid_positions
    else:
      raise NoValidPositionException

def createShelfBowlStackingEnv(config):
  return ShelfBowlStackingEnv(config)
  

if __name__ == '__main__':
  object_init_space = np.asarray([[0.3, 0.7],
                          [-0.4, 0.4],
                          [0, 0.40]])
  env_config = {'object_init_space': object_init_space, 'max_steps': 10, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyzrrr', 'num_objects': 5, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'object_init_space_check': 'point', 'physics_mode': 'slow'}
  planner_config = {'random_orientation': True}

  env = ShelfBowlStackingEnv(env_config)
  planner = ShelfBowlStackingPlanner(env, planner_config)
  s, in_hand, obs = env.reset()
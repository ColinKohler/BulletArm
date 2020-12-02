from copy import deepcopy
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv, NoValidPositionException
from envs.pybullet_envs.ramp_envs.pybullet_tilt_env import PyBulletTiltEnv
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.utils import pybullet_util
import helping_hands_rl_envs.simulators.pybullet.utils.object_generation as pb_obj_generation
import numpy as np

def createTiltImproviseHouseBuilding6Env(simulator_base_env, config):
  class TiltImproviseHouseBuilding6Env(PyBulletTiltEnv):
    def __init__(self, config):
      if simulator_base_env is PyBulletEnv:
        super().__init__(config)
      else:
        raise ValueError('Bad simulator base env specified.')
      self.simulator_base_env = simulator_base_env
      self.random_orientation = config['random_orientation'] if 'random_orientation' in config else False
      self.num_obj = config['num_objects'] if 'num_objects' in config else 1
      self.reward_type = config['reward_type'] if 'reward_type' in config else 'sparse'
      self.tilt_min_dist = 0.03

    def step(self, action):
      self.takeAction(action)
      self.wait(100)
      obs = self._getObservation(action)
      done = self._checkTermination()
      reward = 1.0 if done else 0.0

      if not done:
        done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
      self.current_episode_steps += 1

      return obs, reward, done

    def resetWithTiltAndObj(self, obj_dict):
      while True:
        super().reset()
        self.resetTilt()
        try:
          existing_pos = []
          for t in obj_dict:
            padding = pybullet_util.getPadding(t, self.max_block_size)
            min_distance = pybullet_util.getMinDistance(t, self.max_block_size)
            for j in range(100):
              if t == constants.RANDOM:
                for i in range(100):
                  off_tilt_pos = self._getValidPositions(padding, min_distance, existing_pos, 1)
                  if self.isPosOffTilt(off_tilt_pos[0]):
                    break
                if i == 100:
                  raise NoValidPositionException
                other_pos = self._getValidPositions(padding, min_distance, existing_pos + off_tilt_pos, obj_dict[t] - 1)
                other_pos.extend(off_tilt_pos)
              else:
                other_pos = self._getValidPositions(padding, min_distance, existing_pos, obj_dict[t])
              if all(map(lambda p: self.isPosDistToTiltValid(p, t), other_pos)):
                break

            existing_pos.extend(deepcopy(other_pos))
            orientations = self.calculateOrientations(other_pos)
            if t == constants.RANDOM:
              for i in range(obj_dict[t]):
                zscale = np.random.uniform(2, 2.2)
                scale = np.random.uniform(0.6, 0.9)
                zscale = 0.6 * zscale / scale
                self._generateShapes(t, 1, random_orientation=False, pos=other_pos[i:i+1], rot=orientations[i:i+1], scale=scale, z_scale=zscale)
            elif t == constants.BRICK:
              for i in range(obj_dict[t]):
                brick_xscale = np.random.uniform(0.5, 0.7)
                brick_yscale = np.random.uniform(0.5, 0.7)
                brick_zscale = np.random.uniform(0.4, 0.7)
                handle = pb_obj_generation.generateRandomBrick(other_pos[0],
                                                               orientations[0],
                                                               brick_xscale, brick_yscale, brick_zscale)
                self.objects.append(handle)
                self.object_types[handle] = constants.BRICK
                # self._generateShapes(t, 1, random_orientation=False, pos=other_pos[i:i+1], rot=orientations[i:i+1], scale=scale)

            else:
              self._generateShapes(t, obj_dict[t], random_orientation=False, pos=other_pos, rot=orientations)
        except Exception as e:
          continue
        else:
          break

    def reset(self):
      super().reset()
      obj_dict = {
        constants.ROOF: 1,
        constants.BRICK: 1,
        constants.RANDOM: 2
      }
      self.resetWithTiltAndObj(obj_dict)
      return self._getObservation()

    def _checkTermination(self):
      random_blocks = list(filter(lambda x: self.object_types[x] == constants.RANDOM, self.objects))
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

      top_blocks = []
      for block in random_blocks:
        if self._isObjOnTop(block, random_blocks):
          top_blocks.append(block)
      if len(top_blocks) != 2:
        return False
      if self._checkOnTop(top_blocks[0], bricks[0]) and \
          self._checkOnTop(top_blocks[1], bricks[0]) and \
          self._checkOnTop(bricks[0], roofs[0]) and \
          self._checkOriSimilar([bricks[0], roofs[0]]) and \
          self._checkInBetween(bricks[0], random_blocks[0], random_blocks[1]) and \
          self._checkInBetween(roofs[0], random_blocks[0], random_blocks[1]):
        return True
      return False

  def _thunk():
    return TiltImproviseHouseBuilding6Env(config)

  return _thunk
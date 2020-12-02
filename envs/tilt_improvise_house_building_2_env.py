from copy import deepcopy
import pybullet as pb
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv, NoValidPositionException
from helping_hands_rl_envs.envs.pybullet_tilt_env import PyBulletTiltEnv
from helping_hands_rl_envs.simulators.pybullet.robots.kuka_float_pick import KukaFloatPick
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.utils import pybullet_util
import numpy.random as npr
import numpy as np

def createTiltImproviseHouseBuilding2Env(simulator_base_env, config):
  class TiltImproviseHouseBuilding2Env(PyBulletTiltEnv):
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
                scale = np.random.uniform(0.5, 0.7)
                self._generateShapes(t, 1, random_orientation=False, pos=other_pos[i:i+1], rot=orientations[i:i+1], scale=scale)

            else:
              self._generateShapes(t, obj_dict[t], random_orientation=False, pos=other_pos, rot=orientations)
        except Exception as e:
          continue
        else:
          break

    def reset(self):
      # super().reset()
      obj_dict = {
        constants.ROOF: 1,
        constants.RANDOM: 2
      }
      self.resetWithTiltAndObj(obj_dict)
      return self._getObservation()

    def _checkTermination(self):
      random_blocks = list(filter(lambda x: self.object_types[x] == constants.RANDOM, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

      if self._checkOnTop(random_blocks[0], roofs[0]) and self._checkOnTop(random_blocks[1], roofs[0]):
        return True
      return False


  def _thunk():
    return TiltImproviseHouseBuilding2Env(config)

  return _thunk
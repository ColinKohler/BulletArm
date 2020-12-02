from copy import deepcopy
import pybullet as pb
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv, NoValidPositionException
from helping_hands_rl_envs.envs.pybullet_tilt_env import PyBulletTiltEnv
from helping_hands_rl_envs.simulators.pybullet.robots.kuka_float_pick import KukaFloatPick
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.utils import pybullet_util
import numpy.random as npr
from itertools import combinations
import numpy as np

def createTiltImproviseHouseBuilding3Env(simulator_base_env, config):
  class TiltImproviseHouseBuilding3Env(PyBulletTiltEnv):
    def __init__(self, config):
      if simulator_base_env is PyBulletEnv:
        super().__init__(config)
      else:
        raise ValueError('Bad simulator base env specified.')
      self.simulator_base_env = simulator_base_env
      self.random_orientation = config['random_orientation'] if 'random_orientation' in config else False
      self.num_obj = config['num_objects'] if 'num_objects' in config else 1
      self.reward_type = config['reward_type'] if 'reward_type' in config else 'sparse'
      self.tilt_min_dist = 0.04

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

            orientations = []
            existing_pos.extend(deepcopy(other_pos))
            for position in other_pos:
              y1, y2 = self.getY1Y2fromX(position[0])
              if position[1] > y1:
                d = (position[1] - y1) * np.cos(self.tilt_rz)
                position.append(self.tilt_z1 + 0.02 + np.tan(self.tilt_plain_rx) * d)
                orientations.append(pb.getQuaternionFromEuler([self.tilt_plain_rx, 0, self.tilt_rz]))
              elif position[1] < y2:
                d = (y2 - position[1]) * np.cos(self.tilt_rz)
                position.append(self.tilt_z2 + 0.02 + np.tan(-self.tilt_plain2_rx) * d)
                orientations.append(pb.getQuaternionFromEuler([self.tilt_plain2_rx, 0, self.tilt_rz]))
              else:
                position.append(0.02)
                orientations.append(pb.getQuaternionFromEuler([0, 0, np.random.random() * np.pi * 2]))
            if t == constants.RANDOM:
              for i in range(obj_dict[t]):
                self._generateShapes(t, 1, random_orientation=False, pos=other_pos[i:i+1], rot=orientations[i:i+1], z_scale=npr.choice([constants.z_scale_1, constants.z_scale_2], p=[0.75, 0.25]))
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
        constants.RANDOM: self.num_obj-1
      }
      self.resetWithTiltAndObj(obj_dict)
      return self._getObservation()

    def _checkTermination(self):
      rand_objs = list(filter(lambda x: self.object_types[x] == constants.RANDOM, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
      if roofs[0].getZPosition() < 1.2*self.min_block_size:
        return False

      rand_obj_combs = combinations(rand_objs, 2)
      for (obj1, obj2) in rand_obj_combs:
        if self._checkOnTop(obj1, roofs[0]) and self._checkOnTop(obj2, roofs[0]) and self.isPosOffTilt(obj1.getXYPosition()) and self.isPosOffTilt(obj2.getXYPosition()):
          return True
      return False

  def _thunk():
    return TiltImproviseHouseBuilding3Env(config)

  return _thunk
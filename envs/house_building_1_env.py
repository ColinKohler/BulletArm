from copy import deepcopy
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv

def createHouseBuilding1Env(simulator_base_env, config):
  class HouseBuilding1Env(simulator_base_env):
    ''''''
    def __init__(self, config):
      if 'pick_rot' not in config:
        config['pick_rot'] = True
      if 'place_rot' not in config:
        config['place_rot'] = False
      if 'scale' not in config:
        config['scale'] = 1.
      if 'robot' not in config:
        config['robot'] = 'ur5'
      if 'pos_candidate' not in config:
        config['pos_candidate'] = None
      if 'perfect_grasp' not in config:
        config['perfect_grasp'] = False

      if simulator_base_env is PyBulletEnv:
        super().__init__(config['seed'], config['workspace'], config['max_steps'],
                         config['obs_size'], config['fast_mode'], config['render'],
                         config['action_sequence'], config['simulate_grasp'],
                         config['pos_candidate'], config['perfect_grasp'], config['robot'])
      else:
        raise ValueError('Bad simulator base env specified.')
      self.simulator_base_env = simulator_base_env
      self.random_orientation = config['random_orientation'] if 'random_orientation' in config else False
      self.num_obj = config['num_objects'] if 'num_objects' in config else 1
      self.reward_type = config['reward_type'] if 'reward_type' in config else 'sparse'

      self.blocks = []
      self.triangles = []
      self.stacking_state = {}

    def step(self, action):
      self.takeAction(action)
      self.wait(100)
      obs = self._getObservation()
      done = self._checkTermination()
      curr_num_top = self._getNumTopBlock()
      if self.reward_type == 'step_left':
        reward = self.getStepLeft()
      else:
        reward = 1.0 if done else 0.0

      if not done:
        done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
      self.current_episode_steps += 1

      return obs, reward, done

    def reset(self):
      ''''''
      super(HouseBuilding1Env, self).reset()
      self.triangles = self._generateShapes(self.TRIANGLE, 1, random_orientation=self.random_orientation)
      self.blocks = self._generateShapes(0, self.num_obj-1, random_orientation=self.random_orientation)
      return self._getObservation()

    def saveState(self):
      super(HouseBuilding1Env, self).saveState()
      self.stacking_state = {'blocks': deepcopy(self.blocks),
                             'triangles': deepcopy(self.triangles)}

    def restoreState(self):
      super(HouseBuilding1Env, self).restoreState()
      self.blocks = self.stacking_state['blocks']
      self.triangles = self.stacking_state['triangles']

    def _checkTermination(self):
      ''''''
      # return self._getNumTopBlock() == 1
      return self._checkStack() and self._checkObjUpright(self.triangles[0])

    def getObjectPosition(self):
      return list(map(self._getObjectPosition, self.objects))

    def getPlan(self):
      return self.planHouseBuilding1(self.blocks, self.triangles)

    def getStepLeft(self):
      if not self.isSimValid():
        return 100
      step_left = 2 * (self._getNumTopBlock() - 1)
      if self._isHolding():
        step_left -= 1
        if self._isObjectHeld(self.triangles[0]):
          step_left -= 1
      return step_left

    def isSimValid(self):
      return self._checkObjUpright(self.triangles[0]) and super().isSimValid()


  def _thunk():
    return HouseBuilding1Env(config)

  return _thunk

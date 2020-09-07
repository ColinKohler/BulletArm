import numpy as np
import numpy.random as npr

from helping_hands_rl_envs.planners.base_planner import BasePlanner

class RandomPlanner(BasePlanner):
  def __init__(self, env, config):
    super(RandomPlanner, self).__init__(env, config)

  def getNextAction(self):
    location = npr.uniform(self.env.action_space[:,0], self.env.action_space[:,1])

    if self.env.action_has_primative:
      primative = npr.randint(self.env.num_primatives)
      return np.concatenate(([primative], location))
    else:
      return location

  def getStepLeft(self):
    return 100

import numpy as np
import numpy.random as npr

from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.planners.block_stacking_planner import BlockStackingPlanner
from helping_hands_rl_envs.planners.block_adjacent_planner import BlockAdjacentPlanner

class MultiTaskPlanner(object):
  def __init__(self, env, configs):
    self.env = env

    self.planners = dict()
    for config in configs:
      if config['planner_type'] == 'block_stacking':
        self.planners['block_stacking'] = BlockStackingPlanner(env.envs['block_stacking'], config)
      elif config['planner_type'] == 'block_adjacent':
        self.planners['block_adjacent'] = BlockAdjacentPlanner(env.envs['block_adjacent'], config)

  def getNextAction(self):
    return self.planners[self.env.env_types[self.env.active_env_id]].getNextAction()

  def getStepsLeft(self):
    return self.planners[self.env.env_types[self.env.active_env_id]].getStepsLeft()

  def getValue(self):
    return self.planners[self.env.env_types[self.env.active_env_id]].getValue()

import numpy as np
import numpy.random as npr

from bulletarm.planners.base_planner import BasePlanner
from bulletarm.planners.block_stacking_planner import BlockStackingPlanner
from bulletarm.planners.brick_stacking_planner import BrickStackingPlanner
from bulletarm.planners.pyramid_stacking_planner import PyramidStackingPlanner
from bulletarm.planners.house_building_1_planner import HouseBuilding1Planner
from bulletarm.planners.house_building_2_planner import HouseBuilding2Planner
from bulletarm.planners.house_building_3_planner import HouseBuilding3Planner
from bulletarm.planners.house_building_4_planner import HouseBuilding4Planner
from bulletarm.planners.deconstruct_planner import DeconstructPlanner
from bulletarm.planners.play_planner import PlayPlanner

class MultiTaskPlanner(object):
  def __init__(self, env, configs):
    self.env = env

    self.planners = list()
    for i, config in enumerate(configs):
      if config['planner_type'] == 'block_stacking':
        self.planners.append(BlockStackingPlanner(env.envs[i], config))
      elif config['planner_type'] == 'brick_stacking':
        self.planners.append(BrickStackingPlanner(env.envs[i], config))
      elif config['planner_type'] == 'pyramid_stacking':
        self.planners.append(PyramidStackingPlanner(env.envs[i], config))
      elif config['planner_type'] == 'house_building_1':
        self.planners.append(HouseBuilding1Planner(env.envs[i], config))
      elif config['planner_type'] == 'house_building_2':
        self.planners.append(HouseBuilding2Planner(env.envs[i], config))
      elif config['planner_type'] == 'house_building_3':
        self.planners.append(HouseBuilding3Planner(env.envs[i], config))
      elif config['planner_type'] == 'house_building_4':
        self.planners.append(HouseBuilding4Planner(env.envs[i], config))
      elif config['planner_type'] == 'deconstruct_house_1':
        self.planners.append(DeconstructPlanner(env.envs[i], config))
      elif config['planner_type'] == 'deconstruct_house_2':
        self.planners.append(DeconstructPlanner(env.envs[i], config))
      elif config['planner_type'] == 'deconstruct_house_3':
        self.planners.append(DeconstructPlanner(env.envs[i], config))
      elif config['planner_type'] == 'deconstruct_house_4':
        self.planners.append(DeconstructPlanner(env.envs[i], config))
      elif config['planner_type'] == 'play':
        self.planners.append(PlayPlanner(env.envs[i], config))
      else:
        raise ValueError('Planner type not implemented in Multi-task planner')

  def getRandomAction(self):
    return self.planners[self.active_env_id].getRandomAction()

  def getNextAction(self):
    return self.planners[self.env.active_env_id].getNextAction()

  def getStepsLeft(self):
    return self.planners[self.env.active_env_id].getStepsLeft()

  def getValue(self):
    return self.planners[self.env.active_env_id].getValue()

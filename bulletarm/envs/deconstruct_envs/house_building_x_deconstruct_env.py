import time
from copy import deepcopy
import numpy.random as npr
import numpy as np
from itertools import combinations
from bulletarm.envs.deconstruct_envs.deconstruct_env import DeconstructEnv
from bulletarm.pybullet.utils import constants
from bulletarm.envs.utils.check_goal import CheckGoal
from bulletarm.envs.utils.gen_goal import GenGoal

class HouseBuildingXDeconstructEnv(DeconstructEnv):
  ''''''
  def __init__(self, config):
    super(HouseBuildingXDeconstructEnv, self).__init__(config)
    self.pick_offset = 0.01

    goal = config["goal_string"]
    self.check_goal = CheckGoal(goal, self)

    kwargs = {
      "additional_objects": self.additional_objects
    }

    for key in ["gen_blocks", "gen_bricks", "gen_triangles", "gen_roofs"]:
      if key in config:
        kwargs[key] = config[key]

    self.gen_goal = GenGoal(goal, self, **kwargs)

  def step(self, action):
    reward = 1.0 if self.checkStructure() else 0.0
    self.takeAction(action)
    self.wait(100)
    obs = self._getObservation(action)
    motion_primative, x, y, z, rot = self._decodeAction(action)
    done = motion_primative and self._checkTermination()

    if not done:
      done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
    self.current_episode_steps += 1

    return obs, reward, done

  def reset(self):
    ''''''
    while True:
      super(HouseBuildingXDeconstructEnv, self).reset()
      self.gen_goal.gen()

      while not self.checkStructure():
        super(HouseBuildingXDeconstructEnv, self).reset()
        self.gen_goal.gen()

      return self._getObservation()

  def _checkTermination(self):
    obj_combs = combinations(self.objects, 2)
    for (obj1, obj2) in obj_combs:
      dist = np.linalg.norm(np.array(obj1.getXYPosition()) - np.array(obj2.getXYPosition()))
      if dist < 2.4*self.min_block_size:
        return False
    return True

  def checkStructure(self):
    return self.check_goal.check()

  def isSimValid(self):
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
    if len(roofs) > 0:
      return self._checkObjUpright(roofs[0]) and super(HouseBuildingXDeconstructEnv, self).isSimValid()
    else:
      return super(HouseBuildingXDeconstructEnv, self).isSimValid()

def createHouseBuildingXDeconstructEnv(config):
  return HouseBuildingXDeconstructEnv(config)

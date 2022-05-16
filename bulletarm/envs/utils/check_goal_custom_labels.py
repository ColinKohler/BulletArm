import copy as cp
from bulletarm.envs.utils.check_goal import CheckGoal
from bulletarm.pybullet.utils import constants

class CheckGoalCustomLabels(CheckGoal):
  LABEL_PICK_FAIL = 0
  LABEL_PICK_CUBE = 1
  LABEL_PICK_BRICK = 2
  LABEL_PICK_ROOF = 3
  LABEL_PICK_TRIANGLE = 4

  # TODO: enumerating every possible case by hand works only for a few tasks
  LABEL_PLACE_OTHER = 5
  LABEL_PLACE_TWO_BLOCKS_GROUND = 6
  LABEL_PLACE_ROOF_ON_TWO_BLOCKS = 7
  LABEL_PLACE_BLOCK_ON_BLOCK = 8
  LABEL_PLACE_ROOF_ON_STACK_OF_TWO_BLOCKS = 9

  def __init__(self, goal, env):
    super(CheckGoalCustomLabels, self).__init__(goal, env)

  def get_label_pick(self):
    if self.env.robot.holding_obj is None:
      return self.LABEL_PICK_FAIL

    obj = self.env.object_types[self.env.robot.holding_obj]

    if obj == constants.CUBE:
      return self.LABEL_PICK_CUBE
    elif obj == constants.BRICK:
      return self.LABEL_PICK_BRICK
    elif obj == constants.ROOF:
      return self.LABEL_PICK_ROOF
    elif obj == constants.TRIANGLE:
      return self.LABEL_PICK_TRIANGLE
    else:
      raise ValueError("Holding unknown object.")

  def get_label_place(self, previous_candidates):
    candidates = self.get_place_candidate_dict()

    # TODO: I might be able to do this for any task
    # compare to old candidates
    if self.goal == "1b1b1r":
      if len(previous_candidates["1b1b"][1]) == 0 and len(candidates["1b1b"][1]) > 0:
        # 1b1b previously didn't exist, now it does
        label = self.LABEL_PLACE_BLOCK_ON_BLOCK
      elif len(previous_candidates["1b1b1r"][2]) == 0 and len(candidates["1b1b1r"][2]) > 0:
        # reached goal
        label = self.LABEL_PLACE_ROOF_ON_STACK_OF_TWO_BLOCKS
      else:
        # other place
        label = self.LABEL_PLACE_OTHER
    elif self.goal == "2b2r":
      if len(previous_candidates["2b"][0]) == 0 and len(candidates["2b"][0]) > 0:
        # 2b previously didn't exist, now it does
        label = self.LABEL_PLACE_TWO_BLOCKS_GROUND
      elif len(previous_candidates["2b2r"][1]) == 0 and len(candidates["2b2r"][1]) > 0:
        # reached goal
        label = self.LABEL_PLACE_ROOF_ON_TWO_BLOCKS
      else:
        # other place
        label = self.LABEL_PLACE_OTHER
    else:
      raise NotImplementedError("No place labels for this task.")

    return label, candidates

  def get_place_candidate_dict(self):
    # get new candidates
    candidates = {}

    # TODO: I might be able to do this for any task
    if self.goal == "1b1b1r":
      self.parse_new_goal_and_add_candidates("1b1b", candidates)
      self.parse_new_goal_and_add_candidates("1b1b1r", candidates)
    elif self.goal == "2b2r":
      self.parse_new_goal_and_add_candidates("2b", candidates)
      self.parse_new_goal_and_add_candidates("2b2r", candidates)
    else:
      raise NotImplementedError("No place labels for this task.")

    return candidates

  def parse_new_goal_and_add_candidates(self, goal, candidate_dict):
    self.goal = goal
    self.parse_goal_()
    self.levels.check()

    candidate_dict[goal] = [
      cp.deepcopy(self.levels.levels[i].candidates) for i in range(len(self.levels.levels))
    ]

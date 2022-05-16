import numpy as np
from bulletarm.planners.close_loop_planner import CloseLoopPlanner
from bulletarm.pybullet.utils import transformations

class CloseLoopBlockInBowlPlanner(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)
    self.current_target = None
    self.stage = 0
    self.pre_grasp_pos = self.env.workspace.mean(1)
    self.grasp_pos = self.env.workspace.mean(1)
    self.post_grasp_pos = self.env.workspace.mean(1)
    self.release_pos = self.env.workspace.mean(1)
    self.rot = 0

  def getNextActionToCurrentTarget(self):
    x, y, z, r = self.getActionByGoalPose(self.current_target[0], self.current_target[1])
    if np.all(np.abs([x, y, z]) < self.dpos) and (not self.random_orientation or np.abs(r) < self.drot):
      p = self.current_target[3]
      self.current_target = None
    else:
      p = self.current_target[2]
    return self.env._encodeAction(p, x, y, z, r)

  def setWaypoints(self):
    block_pos = self.env.objects[0].getPosition()
    block_rot = transformations.euler_from_quaternion(self.env.objects[0].getRotation())[2]
    self.rot = block_rot
    bowl_pos = self.env.objects[1].getPosition()
    self.pre_grasp_pos = [block_pos[0], block_pos[1], block_pos[2] + 0.1]
    self.grasp_pos = [block_pos[0], block_pos[1], block_pos[2]]
    self.post_grasp_pos = [block_pos[0], block_pos[1], block_pos[2] + 0.1]
    self.release_pos = [bowl_pos[0], bowl_pos[1], bowl_pos[2] + 0.1]

  def setNewTarget(self):
    if self.stage == 0:
      self.setWaypoints()
      self.current_target = (self.pre_grasp_pos, self.rot, 1, 1)
      self.stage = 1
    elif self.stage == 1:
      self.current_target = (self.grasp_pos, self.rot, 1, 0)
      self.stage = 2
    elif self.stage == 2:
      self.current_target = (self.post_grasp_pos, self.rot, 0, 0)
      self.stage = 3
    elif self.stage == 3:
      self.current_target = (self.release_pos, self.rot, 0, 1)
      self.stage = 0

  def getNextAction(self):
    if self.env.current_episode_steps == 1:
      self.current_target = None
      self.stage = 0
    if self.current_target is not None:
      return self.getNextActionToCurrentTarget()
    else:
      self.setNewTarget()
      return self.getNextActionToCurrentTarget()

  def getStepsLeft(self):
    return 100
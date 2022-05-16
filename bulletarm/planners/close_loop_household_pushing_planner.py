import numpy as np
from bulletarm.planners.close_loop_planner import CloseLoopPlanner

class CloseLoopHouseholdPushingPlanner(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)
    self.direction = 0  # 0-3
    self.iter = 0
    self.max_iter = 5
    self.d = self.env.workspace_size / (self.max_iter+2 - 1)
    self.push_start_pos = self.env.workspace.mean(1)
    self.push_end_pos = self.env.workspace.mean(1)
    self.pre_push_start_pos = self.env.workspace.mean(1)
    self.post_push_end_pos = self.env.workspace.mean(1)
    self.push_rot = 0
    self.current_target = None
    self.stage = 0

  def setWaypoints(self):
    # set pushing waypoints
    if self.direction == 0:
      x = self.env.workspace[0][0]
      y = self.env.workspace[1][0] + (self.iter+1)* self.d
      x_ = x + self.env.workspace_size * 2 / 3
      y_ = y
      r = np.pi / 2
    elif self.direction == 2:
      x = self.env.workspace[0][1]
      y = self.env.workspace[1][0] + (self.iter+1) * self.d
      x_ = x - self.env.workspace_size * 2 / 3
      y_ = y
      r = np.pi / 2
    elif self.direction == 3:
      x = self.env.workspace[0][0] + (self.iter+1) * self.d
      y = self.env.workspace[1][0]
      x_ = x
      y_ = y + self.env.workspace_size * 2 / 3
      r = 0
    else:
      x = self.env.workspace[0][0] + (self.iter+1) * self.d
      y = self.env.workspace[1][1]
      x_ = x
      y_ = y - self.env.workspace_size * 2 / 3
      r = 0

    self.push_start_pos = (x, y, self.env.workspace[2][0])
    self.pre_push_start_pos = (x, y, self.env.workspace[2][0] + 0.1)
    self.push_end_pos = (x_, y_, self.env.workspace[2][0])
    self.post_push_end_pos = (x_, y_, self.env.workspace[2][0] + 0.1)
    self.push_rot = r

  def setNewTarget(self):
    if self.stage == 0:
      self.setWaypoints()
      # to pre push start pos
      self.current_target = (self.pre_push_start_pos, self.push_rot, 0.5)
      self.stage = 1
    elif self.stage == 1:
      # to push start pos
      self.current_target = (self.push_start_pos, self.push_rot, 0.5)
      self.stage = 2
    elif self.stage == 2:
      # to push end pos
      self.current_target = (self.push_end_pos, self.push_rot, 0.5)
      self.stage = 3
    elif self.stage == 3:
      # to post end pos
      self.current_target = (self.post_push_end_pos, self.push_rot, 0.5)
      self.stage = 0
      self.iter += 1
      self.iter %= self.max_iter

  def getNextActionToCurrentTarget(self):
    x, y, z, r = self.getActionByGoalPose(self.current_target[0], self.current_target[1])
    primitive = self.current_target[2]
    if np.all(np.abs([x, y, z]) < self.dpos) and np.abs(r) < self.drot:
      self.current_target = None
    return np.array([primitive, x, y, z, r])

  def getNextAction(self):
    if self.env.current_episode_steps == 1:
      # bin_center = self.env.bin0_center if self.env.bin_id == 0 else self.env.bin1_center
      # depth = self.env.getHeightmapReconstruct(gripper_pos=[*bin_center, 0.2 + self.env.z_min])
      # # depth = extractRec(depth, [x_pixel, y_pixel], gripper_rot)
      # depth_shape = depth.shape
      # depth = -depth
      # heightmap = depth + 0.2
      # num_points = []
      # for i in range(4):
      #   heightmap = np.rot90(heightmap)
      #   check_area = heightmap[:, :int(depth_shape[1] / 2)]
      #   num_points.append((check_area > 0.02).sum())
      # self.direction = np.argmin(num_points)
      self.direction = np.random.randint(4)
      self.iter = 0
      self.stage = 0
      self.current_target = None
    if self.current_target is not None:
      return self.getNextActionToCurrentTarget()
    else:
      self.setNewTarget()
      return self.getNextActionToCurrentTarget()

  def getStepsLeft(self):
    return 100
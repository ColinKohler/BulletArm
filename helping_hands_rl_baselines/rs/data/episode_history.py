import copy
import numpy as np
import matplotlib.pyplot as plt

from data import data_utils

class EpisodeHistory(object):
  def __init__(self, expert_traj=False):
    self.expert_traj = expert_traj
    self.obs_history = list()
    self.q_map_history = list()
    self.pred_obs_history = list()
    self.action_history = list()
    self.sampled_action_history = list()
    self.reward_history = list()
    self.value_history = list()
    self.child_visits = list()

    self.priorities = None
    self.eps_priority = None

class Node(object):
  def __init__(self, parent, depth, obs, value, reward, q_map=None):
    self.parent = parent
    self.children = dict()
    self.children_values = dict()
    self.obs = obs
    self.value = value
    self.reward = reward
    self.depth = depth
    self.q_map = q_map
    self.sampled_actions = list()

  def expanded(self):
    return len(self.children) > 0

  def expand(self, actions, state_, hand_obs_, obs_, values_, reward_, q_map):
    for i, action in enumerate(actions):
      action = tuple(action.tolist())
      self.children[action] = Node((self, action),
                                    self.depth+1,
                                    (state_[i], hand_obs_[i], obs_[i]),
                                    values_[i].item(),
                                    reward_[i].item(),
                                    q_map[i].cpu().numpy())
      self.children_values[action] = values_[i].item()

  def isTerminal(self):
    return self.reward >= 1.

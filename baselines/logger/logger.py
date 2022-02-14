'''
.. moduleauthor: Colin Kohler <github.com/ColinKohler>
'''

import os
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Logger(object):
  '''
  Logger class. Writes log data to tensorboard.

  Args:
    results_path (str): Path to save log files to
    hyperparameters (dict): Hyperparameters to log. Defaults to None
  '''
  def __init__(self, results_path, hyperpameters=None):
    self.results_path = results_path
    self.writer = SummaryWriter(results_path)
    self.scalar_logs = dict()

    self.num_steps = 0
    self.num_eps = 0
    self.training_step = 0
    self.rewards = list()
    self.current_episode_rewards = list()

    if hyperparameters:
      hp_table = [
        f'| {k} | {v} |' for k, v in hyperpameters.items()
      ]
      self.writer.add_text(
        'Hyperparameters', '| Parameter | Value |\n|-------|-------|\n' + '\n'.join(hp_table)
      )

  def logStep(self, rewards, done_masks):
    '''
    Log episode step.

    Args:
      rewards (list[double]): List of rewards
      done_masks (list[int]):
    '''
    if self.current_episode_rewards and len(rewards) != len(self.current_episode_rewards):
      raise ValueError("Length of rewards different than was previously logged.")

    self.num_steps += len(rewards)
    self.num_eps += np.sum(done_masks)
    for i, (reward, done) in enumerate(zip(rewards, done_masks)):
      if done:
        self.rewards.append(current_episode_rewards[i] + reward)
      else:
        self.current_episode_rewards[i] += reward

  def logEpsode(self, rewards):
    '''
    Log a episode.

    Args:
      rewards (list[double]: Rewards for the entire episode
    '''
    self.num_steps += int(len(rewards))
    self.num_eps += 1
    self.rewards.append(np.sum(rewards))

  def writeLog(self):
    '''
    Write the logdir to the tensorboard summary writer. Calling this too often can
    slow down training.
    '''
    writer.add_scalar('1.Eval/1.Learning_curve',
                      np.mean(self.rewards[-100:]) if self.rewards else 0,
                      self.log_counter)

    writer.add_scalar('2.Data/1.Num_eps', self.num_eps, self.log_counter)
    writer.add_scalar('2.Data/2.Num_steps', self.num_steps, self.log_counter)

    for k, v in self.scalar_logs.items():
      writer.add_scalar(k, v, self.log_counter)

    self.log_counter += 1

  def exportData(self, filepath):
    '''
    Export log data as a pickle

    Args:
      filepath (str): The filepath to save the exported data to
    '''
    pickle.dump(
      {
        'num_eps' : self.num_eps,
        'num_steps' : self.num_steps,
        'num_training_steps' : self.training_step,
        'rewards' : self.rewards,
      },
      open(os.path.join(self.results_path, 'log_data.pkl'), 'wb')
    )

  def getScalars(self, keys):
    '''
    Get data from the scalar log dict.

    Args:
      keys (str | list[str]): Key or list of keys to get from the scalar log dict

    Returns:
      Object | Dict: Single object when single key is passed or dict containing objects from
        all keys
    '''
    if isinstance(keys, str):
      return self.scalar_logs[keys]
    elif isinstance(keys, list):
      return {key: self.scalar_logs[key] for key in keys}
    else:
      raise TypeError

  def updateScalars(self, keys, values=None):
    '''
    Update the scalar log dict with new values.

    Args:
      key (str | dict): Either key to update to the given value or a collection of key-value
        pairs to update
      value (Object): Value to update the key to in the log dir. Defaults to None
    '''
    if isinstance(keys, str) and values is not None:
      self.scalar_logs[keys] = values
    elif isinstance(key, dict):
      self.scalar_logs.update(keys)
    else:
      raise TypeError

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
    num_eval_eps (int): Number of episodes in a evaluation iteration
    hyperparameters (dict): Hyperparameters to log. Defaults to None
  '''
  def __init__(self, results_path, num_eval_eps=100, hyperparameters=None):
    self.results_path = results_path
    self.writer = SummaryWriter(results_path)
    self.log_counter = 0
    self.scalar_logs = dict()

    # Training
    self.num_steps = 0
    self.num_eps = 0
    self.num_training_steps = 0
    self.training_eps_rewards = list()
    self.loss = dict()

    # Evaluation
    self.num_eval_episodes = num_eval_eps # TODO: Dunno if I want this here
    self.num_eval_intervals = 0
    self.eval_eps_rewards = list()
    self.eval_mean_values = list()
    self.eval_eps_lens = list()

    if hyperparameters:
      hp_table = [
        f'| {k} | {v} |' for k, v in hyperparameters.items()
      ]
      self.writer.add_text(
        'Hyperparameters', '| Parameter | Value |\n|-------|-------|\n' + '\n'.join(hp_table)
      )

  # TODO: I don't use this atm so tihs is untested.
  def logStep(self, rewards, done_masks):
    '''
    Log episode step.

    Args:
      rewards (list[float]): List of rewards
      done_masks (list[int]):
    '''
    if self.current_episode_rewards and len(rewards) != len(self.current_episode_rewards):
      raise ValueError("Length of rewards different than was previously logged.")

    self.num_steps += len(rewards)
    self.num_eps += np.sum(done_masks)
    for i, (reward, done) in enumerate(zip(rewards, done_masks)):
      if done:
        self.training_eps_rewards.append(current_episode_rewards[i] + reward)
      else:
        self.current_episode_rewards[i] += reward

  def logTrainingEpisode(self, rewards):
    '''
    Log a episode.

    Args:
      rewards (list[float]: Rewards for the entire episode
    '''
    self.num_steps += int(len(rewards))
    self.num_eps += 1
    self.training_eps_rewards.append(np.sum(rewards))

  def logEvalInterval(self):
    self.num_eval_intervals += 1

  def logEvalEpisode(self, rewards, values):
    '''
    Log a evaulation episode.

    Args:
      rewards (list[float]: Rewards for the episode
      values (list[float]): Values for the episode
    '''
    self.eval_eps_rewards.append(np.sum(rewards))
    self.eval_mean_values.append(np.sum(values))
    self.eval_eps_lens.append(int(len(rewards)))

  def logTrainingStep(self, loss):
    ''''''
    self.num_training_steps += 1

    # TODO: Unsure if this is the best way to handle loss. See loss comment in writeLog().
    for k, v in loss.items():
      if k in self.loss.keys():
        self.loss[k].append(v)
      else:
        self.loss[k] = [v]

  def writeLog(self):
    '''
    Write the logdir to the tensorboard summary writer. Calling this too often can
    slow down training.
    '''

    self.writer.add_scalar('1.Evaluate/1.Reward',
                           self.getAvg(self.eval_eps_rewards, n=self.num_eval_episodes),
                           self.num_eval_intervals)
    self.writer.add_scalar('1.Evaluate/2.Mean_value',
                           self.getAvg(self.eval_mean_values, n=self.num_eval_episodes),
                           self.num_eval_intervals)
    self.writer.add_scalar('1.Evaluate/3.Eps_len',
                          self.getAvg(self.eval_eps_lens, n=self.num_eval_episodes),
                          self.num_eval_intervals)
    # TODO: Do we want to allow custom windows here?
    self.writer.add_scalar('1.Evaluate/4.Learning_curve',
                           self.getAvg(self.training_eps_rewards, n=100),
                           len(self.training_eps_rewards))

    self.writer.add_scalar('2.Data/1.Num_eps', self.num_eps, self.log_counter)
    self.writer.add_scalar('2.Data/2.Num_steps', self.num_steps, self.log_counter)
    self.writer.add_scalar('2.Data/3.Training_steps', self.num_training_steps, self.log_counter)
    self.writer.add_scalar('2.Data/4.Training_steps_per_eps_step_ratio',
                           self.num_training_steps / max(1, self.num_steps),
                           self.log_counter)

    # TODO: This imposes a restriction on the loss dict keys. Dunno if its better to
    #       just give the user free reign in how they want to handle this? There *should*
    #       always be some loss while training though...
    if self.loss:
      for i, (k, v) in enumerate(self.loss.items()):
        self.writer.add_scalar('3.Loss/{}.{}_loss'.format(i+1, k), v[-1], self.log_counter)

    # TODO: I have not needed to test this yet. Can't imagine it doesn't work but still...
    for k, v in self.scalar_logs.items():
      self.writer.add_scalar(k, v, self.log_counter)

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
        'num_training_steps' : self.num_training_steps,
        'training_eps_rewards' : self.rewards,
        'num_eval_intervals' : self.num_eval_intervals,
        'eval_eps_rewards' : self.eval_eps_rewards,
        'eval_mean_values' : self.eval_mean_values,
        'eval_eps_lens' : self.eval_eps_lens,
        'loss' : self.loss,
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

  def getAvg(self, l, n=0):
    '''
    Numpy mean wrapper to handle empty lists.

    Args:
      l (list[float]): The list
      n (int): Number of trail elements to average over. Defaults to entire list.

    Returns:
      float: List average
    '''
    avg = np.mean(l[-n:]) if l else 0
    return avg

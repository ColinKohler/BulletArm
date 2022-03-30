'''
.. moduleauthor: Colin Kohler <github.com/ColinKohler>
'''

import os
import json
import time
import pickle
import torch
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
  def __init__(self, results_path, checkpoint_interval=500, num_eval_eps=100, hyperparameters=None):
    self.results_path = results_path
    self.writer = SummaryWriter(results_path)
    self.checkpoint_interval = checkpoint_interval
    self.log_counter = 0
    self.scalar_logs = dict()

    # Training
    self.num_steps = 0
    self.num_eps = 0
    self.num_training_steps = 0
    self.training_eps_rewards = list()
    self.loss = dict()
    self.current_episode_rewards = None

    # Evaluation
    self.num_eval_episodes = num_eval_eps # TODO: Dunno if I want this here
    self.num_eval_intervals = 0
    self.eval_eps_rewards = [[]]
    self.eval_eps_dis_rewards = [[]]
    self.eval_mean_values = [[]]
    self.eval_eps_lens = [[]]

    # sub folders for saving the models and checkpoint
    self.models_dir = os.path.join(self.results_path, 'models')
    self.checkpoint_dir = os.path.join(self.results_path, 'checkpoint')
    os.makedirs(self.models_dir)
    os.makedirs(self.checkpoint_dir)

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
    if self.current_episode_rewards is None:
      self.current_episode_rewards = [0 for _ in rewards]
    if self.current_episode_rewards and len(rewards) != len(self.current_episode_rewards):
      raise ValueError("Length of rewards different than was previously logged.")

    self.num_steps += len(rewards)
    self.num_eps += np.sum(done_masks)
    for i, (reward, done) in enumerate(zip(rewards, done_masks)):
      if done:
        self.training_eps_rewards.append(self.current_episode_rewards[i] + reward)
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
    self.eval_eps_rewards.append([])
    self.eval_eps_dis_rewards.append([])
    self.eval_mean_values.append([])
    self.eval_eps_lens.append([])

  def logEvalEpisode(self, rewards, values=None, discounted_return=None):
    '''
    Log a evaluation episode.

    Args:
      rewards (list[float]: Rewards for the episode
      values (list[float]): Values for the episode
      discounted_return (list[float]): Discounted return of the episode
    '''
    self.eval_eps_rewards[self.num_eval_intervals].append(np.sum(rewards))
    self.eval_eps_lens[self.num_eval_intervals].append(int(len(rewards)))
    if values is not None:
      self.eval_mean_values[self.num_eval_intervals].append(np.mean(values))
    if discounted_return is not None:
      self.eval_eps_dis_rewards[self.num_eval_intervals].append(discounted_return)

  def logTrainingStep(self, loss):
    ''''''
    self.num_training_steps += 1
    if type(loss) is list or type(loss) is tuple:
      loss = {'loss{}'.format(i): loss[i] for i in range(len(loss))}
    elif type(loss) is float:
      loss = {'loss': loss}
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
    # the eval list index at self.num_eval_intervals might be being updated, so log the eval list indexed
    # at self.num_eval_intervals-1
    if self.num_eval_intervals > 1:
      self.writer.add_scalar('1.Evaluate/1.Reward',
                             self.getAvg(self.eval_eps_rewards[self.num_eval_intervals-1], n=self.num_eval_episodes),
                             self.num_eval_intervals-1)
      self.writer.add_scalar('1.Evaluate/2.Mean_value',
                             self.getAvg(self.eval_mean_values[self.num_eval_intervals-1], n=self.num_eval_episodes),
                             self.num_eval_intervals-1)
      self.writer.add_scalar('1.Evaluate/3.Eps_len',
                            self.getAvg(self.eval_eps_lens[self.num_eval_intervals-1], n=self.num_eval_episodes),
                            self.num_eval_intervals-1)
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

    if self.loss:
      for i, (k, v) in enumerate(self.loss.items()):
        self.writer.add_scalar('3.Loss/{}.{}_loss'.format(i+1, k), v[-1], self.log_counter)

    # TODO: I have not needed to test this yet. Can't imagine it doesn't work but still...
    for k, v in self.scalar_logs.items():
      self.writer.add_scalar(k, v, self.log_counter)

    if self.num_training_steps > 0 and self.num_training_steps % self.checkpoint_interval == 0:
      self.exportData()

    self.log_counter += 1

  def getSaveState(self):
    state = {
      'num_steps': self.num_steps,
      'num_eps': self.num_eps,
      'num_training_steps': self.num_training_steps,
      'training_eps_rewards': self.training_eps_rewards,
      'loss': self.loss,
      'num_eval_intervals': self.num_eval_intervals,
      'eval_eps_rewards': self.eval_eps_rewards,
      'eval_eps_dis_rewards': self.eval_eps_dis_rewards,
      'eval_mean_values': self.eval_mean_values,
      'eval_eps_lens': self.eval_eps_lens,
    }
    return state

  def exportData(self):
    '''
    Export log data as a pickle

    Args:
      filepath (str): The filepath to save the exported data to
    '''
    pickle.dump(
      self.getSaveState(),
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
    elif isinstance(keys, dict):
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

  def saveParameters(self, parameters):
    '''
    Save the parameters as a json file

    Args:
      parameters: parameter dict to save
    '''
    class NumpyEncoder(json.JSONEncoder):
      def default(self, obj):
        if isinstance(obj, np.ndarray):
          return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    with open(os.path.join(self.results_path, "parameters.json"), 'w') as f:
      json.dump(parameters, f, cls=NumpyEncoder)

  def saveCheckPoint(self, agent_save_state, buffer_save_state):
    '''
    Save the checkpoint

    Args:
      agent_save_state (dict): the agent's save state for checkpointing
      buffer_save_state (dict): the buffer's save state for checkpointing
    '''
    checkpoint = {
      'agent': agent_save_state,
      'buffer_state': buffer_save_state,
      'logger': self.getSaveState(),
      'torch_rng_state': torch.get_rng_state(),
      'torch_cuda_rng_state': torch.cuda.get_rng_state(),
      'np_rng_state': np.random.get_state()
    }
    torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'checkpoint.pt'))

  def loadCheckPoint(self, checkpoint_dir, agent_load_func, buffer_load_func):
    '''
    Load the checkpoint

    Args:
      checkpoint_dir: the directory of the checkpoint to load
      agent_load_func (func): the agent's loading checkpoint function. agent_load_func must take a dict as input to
        load the agent's checkpoint
      buffer_load_func (func): the buffer's loading checkpoint function. buffer_load_func must take a dict as input to
        load the buffer's checkpoint
    '''
    print('loading checkpoint')

    checkpoint = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'))

    agent_load_func(checkpoint['agent'])
    buffer_load_func(checkpoint['buffer_state'])

    self.num_steps = checkpoint['logger']['num_steps']
    self.num_eps = checkpoint['logger']['num_eps']
    self.num_training_steps = checkpoint['logger']['num_training_steps']
    self.training_eps_rewards = checkpoint['logger']['training_eps_rewards']
    self.loss = checkpoint['logger']['loss']
    self.num_eval_intervals = checkpoint['logger']['num_eval_intervals']
    self.eval_eps_rewards = checkpoint['logger']['eval_eps_rewards']
    self.eval_eps_dis_rewards = checkpoint['logger']['eval_eps_dis_rewards']
    self.eval_mean_values = checkpoint['logger']['eval_mean_values']
    self.eval_eps_lens = checkpoint['logger']['eval_eps_lens']

    torch.set_rng_state(checkpoint['torch_rng_state'])
    torch.cuda.set_rng_state(checkpoint['torch_cuda_rng_state'])
    np.random.set_state(checkpoint['np_rng_state'])

  def getCurrentLoss(self, n=100):
    '''
    Calculate the average loss of previous n steps
    Args:
      n: the number of previous training steps to calculate the average loss
    Returns:
      the average loss value
    '''
    avg_losses = []
    for k, v in self.loss.items():
      avg_losses.append(self.getAvg(v, n))
    return np.mean(avg_losses)


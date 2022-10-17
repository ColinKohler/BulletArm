'''
.. moduleauthor: Colin Kohler <github.com/ColinKohler>
'''

import os
import pickle
import numpy as np
import more_itertools
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

class Plotter(object):
  '''
  Plotting utility.
  '''
  def __init__(self, log_filepaths, log_names):
    self.logs = self.loadLogs(log_filepaths, log_names)

  def loadLogs(self, filepaths, names):
    '''

    '''
    logs = dict()

    for n, fp in zip(names, filepaths):
      if os.path.exists(fp):
        with open(fp, 'rb') as f:
          logs[n] = pickle.load(f)
      else:
        print('No log found at {}'.format(fp))

    return logs

  def plotLearningCurves(self, title, filepath, window=100, max_eps=None):
    '''
    Plot mulitple learning curves on a single plot.

    Args:
    '''

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Avg. Reward')

    for log_name, log in self.logs.items():
      eps_rewards = log['training_eps_rewards']
      if max_eps:
        eps_rewards = eps_rewards[:max_eps]
      avg_reward = np.mean(list(more_itertools.windowed(eps_rewards, window)), axis=1)
      xs = np.arange(window, len(avg_reward) + window)
      ax.plot(xs, avg_reward, label=log_name)

    ax.legend()
    plt.savefig(filepath)
    plt.close()

  def plotEvalReturns(self, title, filepath, window=1, num_eval_intervals=None, eval_interval=500):
    '''
    Plot mulitple evaluation curves on a single plot.

    Args:
    '''

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Discounted Return')

    for log_name, log in self.logs.items():
      eval_returns = [np.mean(returns) for returns in log['eval_eps_dis_rewards']]
      if num_eval_intervals:
        eval_returns = eval_returns[:num_eval_intervals]
      eval_returns = np.mean(list(more_itertools.windowed(eval_returns, window)), axis=1)
      xs = np.arange(window, len(eval_returns) + window) * eval_interval
      ax.plot(xs, eval_returns, label=log_name)

    ax.legend()
    plt.savefig(filepath)
    plt.close()

  def plotEvalRewards(self, title, filepath, window=1, num_eval_intervals=None, eval_interval=500):
    '''
    Plot mulitple evaluation curves on a single plot.

    Args:
    '''

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Avg. Reward')

    for log_name, log in self.logs.items():
      eval_rewards = [np.mean(rewards) for rewards in log['eval_eps_rewards']]
      if num_eval_intervals:
        eval_rewards = eval_rewards[:num_eval_intervals]
      eval_rewards = np.mean(list(more_itertools.windowed(eval_rewards, window)), axis=1)
      eval_rewards = np.clip(eval_rewards, -1, 1)
      xs = np.arange(window, len(eval_rewards) + window) * eval_interval
      ax.plot(xs, eval_rewards, label=log_name)

    ax.legend()
    plt.savefig(filepath)
    plt.close()

  def plotEvalLens(self, title, filepath, window=1):
    '''
    Plot mulitple evaluation curves on a single plot.

    Args:
    '''

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Avg. Eps Len')

    for log_name, log in self.logs.items():
      eval_lens = [np.mean(lens) for lens in log['eval_eps_lens']]
      eval_lens = np.mean(list(more_itertools.windowed(eval_lens, window)), axis=1)
      xs = np.arange(window, len(eval_lens) + window) * 500
      ax.plot(xs, eval_lens, label=log_name)

    ax.legend()
    plt.savefig(filepath)
    plt.close()

  def plotEvalValues(self, title, filepath, window=1):
    '''
    Plot mulitple evaluation curves on a single plot.

    Args:
    '''

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Avg. Eps Value')

    for log_name, log in self.logs.items():
      eval_values = [np.mean(lens) for lens in log['eval_mean_values']]
      eval_values = np.mean(list(more_itertools.windowed(eval_values, window)), axis=1)
      xs = np.arange(window, len(eval_values) + window) * 500
      ax.plot(xs, eval_values, label=log_name)

    ax.legend()
    plt.savefig(filepath)
    plt.close()

  def plotLearningCurve(self, name, title, filepath, window=100):
    '''
    Plot the learning curve for the given episode rewards.

    Args:
    '''
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Avg. Reward')

    eps_reward = self.logs[name]['training_eps_rewards']
    avg_reward = np.mean(list(more_itertools.windowed(eps_rewards, window)), axis=1)
    xs = np.arange(window, len(avg_reward) + window)
    ax.plot(xs, avg_reward)

    plt.savefig(filepath)
    plt.close()

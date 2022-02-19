'''
.. moduleauthor: Colin Kohler <github.com/ColinKohler>
'''

import numpy as np
import more_itertools
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

class Plotter(object):
  '''
  Plotting utility.
  '''
  def __init__(self):
    pass

  def plotLearningCurves(self, eps_rewards, title, legend_titles, filepath, window=100):
    '''
    Plot mulitple learning curves on a single plot.

    Args:
      eps_rewards ():
      title (str):
      legend_titles (list[str]):
      filepath (str):
      window (int):
    '''
    pass

  def plotLearningCurve(self, eps_rewards, title, filepath, window=100):
    '''
    Plot the learning curve for the given episode rewards.

    Args:
      eps_rewards (list[double]): Episode rewards to plot
      title (str): Figure title
      filepath (str): Filepath to save the plot to
      window (int): Window to average episode rewards over. Default: 100
    '''
    avg_reward = np.mean(list(more_itertools.windowed(eps_rewards, window)), axis=1)
    xs = np.arange(window, len(avg_reward) + window)

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Avg. Reward')

    ax.plot(xs, avg_reward)
    plt.savefig(filepath)
    plt.close()

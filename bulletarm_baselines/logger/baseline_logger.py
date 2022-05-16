import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from more_itertools import windowed

from.logger import Logger

class BaselineLogger(Logger):
  def __init__(self, results_path, checkpoint_interval=500, num_eval_eps=100, hyperparameters=None, eval_freq=100):
    super().__init__(results_path, checkpoint_interval, num_eval_eps, hyperparameters)
    self.info_dir = os.path.join(self.results_path, 'info')
    os.makedirs(self.info_dir)
    self.eval_freq = eval_freq

  def saveLearningCurve(self, n=100):
    ''' Plot the rewards over timesteps and save to logging dir '''
    n = min(n, len(self.training_eps_rewards))
    if n > 0:
      avg_reward = np.mean(list(windowed(self.training_eps_rewards, n)), axis=1)
      xs = np.arange(n, (len(avg_reward)) + n)
      plt.plot(xs, np.mean(list(windowed(self.training_eps_rewards, n)), axis=1))
      plt.savefig(os.path.join(self.info_dir, 'learning_curve.pdf'))
      plt.close()

  def saveLossCurve(self, n=100):
    losses = np.array(list(self.loss.values()))
    losses = np.moveaxis(losses, 0, 1)
    if len(losses) < n:
      return
    if len(losses.shape) == 1:
      losses = np.expand_dims(losses, 0)
    else:
      losses = np.moveaxis(losses, 1, 0)
    for loss in losses:
      plt.plot(np.mean(list(windowed(loss, n)), axis=1))

    plt.savefig(os.path.join(self.info_dir, 'loss_curve.pdf'))
    plt.yscale('log')
    plt.savefig(os.path.join(self.info_dir, 'loss_curve_log.pdf'))

    plt.close()

  def saveEvalCurve(self):
    if len(self.eval_eps_rewards) > 1:
      eval_data = []
      for i in range(len(self.eval_eps_rewards)-1):
        eval_data.append(np.mean(self.eval_eps_rewards[i]))
      xs = np.arange(self.eval_freq, len(self.eval_eps_rewards) * self.eval_freq, self.eval_freq)
      plt.plot(xs, eval_data)
      plt.savefig(os.path.join(self.info_dir, 'eval_curve.pdf'))
      plt.close()

  def saveRewards(self):
    np.save(os.path.join(self.info_dir, 'rewards.npy'), self.training_eps_rewards)

  def saveLosses(self):
    np.save(os.path.join(self.info_dir, 'losses.npy'), self.loss)

  def saveEvalRewards(self):
    np.save(os.path.join(self.info_dir, 'eval_rewards.npy'), np.array(self.eval_eps_rewards, dtype=object))

  def exportData(self):
    super().exportData()
    self.saveLearningCurve()
    self.saveEvalCurve()
    self.saveLossCurve()
    self.saveRewards()
    self.saveEvalRewards()
    self.saveLosses()
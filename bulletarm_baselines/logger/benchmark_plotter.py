'''
.. moduleauthor: Colin Kohler <github.com/ColinKohler>
import os
'''

import os
import pickle
import numpy as np
import more_itertools
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

class BenchmarkPlotter(object):
  '''
  Plotting utility used to plot average multiple runs.
  '''
  def __init__(self, log_info):
    self.logs = self.loadLogs(log_info)

  def loadLogs(self, log_info):
    logs = dict()

    for l in log_info:
      runs = list()
      for fp in l['log_filepaths']:
        if os.path.exists(fp):
          with open(fp, 'rb') as f:
            try:
              runs.append(pickle.load(f))
            except:
              print('File failed to load: {}'.format(fp))
        else:
            print('No log found at {}'.format(fp))

      if l['domain'] in logs.keys():
        logs[l['domain']][l['legend_name']] = runs
      else:
        logs[l['domain']] = {l['legend_name'] : runs}

    return logs

  def plotEvalRewards(self, filepath, grid_shape, num_eval_intervals, window=1, eval_interval=500, process_labels=True):
    num_domains = len(self.logs.keys())

    fig, ax = plt.subplots(nrows=grid_shape[0], ncols=grid_shape[1], figsize=(22,18))

    colors = ['#C44E52', '#55A868', '#4C72B0', '#8C8C8C'][::-1]

    for ax_id, domain in enumerate(self.logs.keys()):
      for j, (log_name, log) in enumerate(self.logs[domain].items()):
        sr = list()
        for run in log:
          eval_rewards = run['eval_eps_rewards']
          if len(eval_rewards) < 10:
            continue

          if num_eval_intervals:
            eval_rewards = eval_rewards[:num_eval_intervals]
          eval_rewards = [np.mean(eps) for eps in eval_rewards]
          avg_eval_rewards = np.mean(list(more_itertools.windowed(eval_rewards, window)), axis=1)
          for i, s in enumerate(avg_eval_rewards):
            if np.isnan(s):
              avg_eval_rewards[i] = avg_eval_rewards[i-1]
          sr.append(avg_eval_rewards)

        max_len = max([s.size for s in sr])
        for i, s in enumerate(sr):
          if s.size < max_len:
            sr[i] = np.pad(s, (0, max_len - s.size), 'edge')
        sr = np.array(sr) * 100
        x = (np.arange(1, sr.shape[1]+1) * eval_interval) / 1e4

        sr_mean = np.mean(sr, axis=0)
        sr_std = np.std(sr, axis=0) / np.sqrt(len(log))

        row = int(ax_id / grid_shape[1])
        col = int(ax_id % grid_shape[1])
        ax[row][col].set_title(' '.join(domain.split('_')).title(), fontsize=14, weight='bold')
        ax[row][col].set_xlabel('Training Steps - 1e4', fontsize=12)
        ax[row][col].set_ylabel('Success Rate (%)', fontsize=12)
        ax[row][col].set_ylim([0, 105])
        if process_labels:
          label = '+'.join(log_name.split('_')).title()
        else:
          label = log_name
        ax[row][col].plot(x, sr_mean.squeeze(), label=label, color=colors[j])
        ax[row][col].fill_between(x, sr_mean.squeeze() - sr_std, sr_mean.squeeze() + sr_std, alpha=0.5, color=colors[j])

    handles, labels = ax[row][col].get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.33, wspace=0.36)
    ax[2][1].legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.12) , fancybox=True, shadow=True, ncol=4)
    plt.tight_layout(pad=2.0)
    plt.savefig(filepath)
    plt.close()

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

    self.eval_freq = eval_freq
    self.info_dir = os.path.join(self.results_path, 'info')
    self.models_dir = os.path.join(self.results_path, 'models')
    self.checkpoint_dir = os.path.join(self.results_path, 'checkpoint')
    os.makedirs(self.info_dir)
    os.makedirs(self.models_dir)
    os.makedirs(self.checkpoint_dir)

  def saveParameters(self, parameters):
    class NumpyEncoder(json.JSONEncoder):
      def default(self, obj):
        if isinstance(obj, np.ndarray):
          return obj.tolist()
        return json.JSONEncoder.default(self, obj)

    with open(os.path.join(self.info_dir, "parameters.json"), 'w') as f:
      json.dump(parameters, f, cls=NumpyEncoder)

  def getCurrentLoss(self, n=100):
    avg_losses = []
    for k, v in self.loss.items():
      avg_losses.append(self.getAvg(v, n))
    return np.mean(avg_losses)

  def saveModel(self, name, agent):
    '''
    Save PyTorch model to log directory

    Args:
      - iteration: Interation of the current run
      - name: Name to save model as
      - agent: Agent containing model to save
    '''
    agent.saveModel(os.path.join(self.models_dir, 'snapshot_{}'.format(name)))

  def saveBuffer(self, buffer):
    print('saving buffer')
    torch.save(buffer.getSaveState(), os.path.join(self.checkpoint_dir, 'buffer.pt'))

  def loadBuffer(self, buffer, path, max_n=1000000):
    print('loading buffer: ' + path)
    load = torch.load(path)
    for i in range(len(load['storage'])):
      if i == max_n:
        break
      t = load['storage'][i]
      buffer.add(t)

  def saveCheckPoint(self, args, agent, buffer):
    checkpoint = {
      'args': args.__dict__,
      'agent': agent.getSaveState(),
      'buffer_state': buffer.getSaveState(),
      'logger': {
        'num_eps' : self.num_eps,
        'num_steps' : self.num_steps,
        'num_training_steps' : self.num_training_steps,
        'training_eps_rewards' : self.training_eps_rewards,
        'num_eval_intervals' : self.num_eval_intervals,
        'eval_eps_rewards' : self.eval_eps_rewards,
        'eval_mean_values' : self.eval_mean_values,
        'eval_eps_lens' : self.eval_eps_lens,
        'loss' : self.loss,
      },
      'torch_rng_state': torch.get_rng_state(),
      'torch_cuda_rng_state': torch.cuda.get_rng_state(),
      'np_rng_state': np.random.get_state()
    }
    if hasattr(agent, 'his'):
      checkpoint.update({'agent_his': agent.his})
    torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'checkpoint.pt'))

  def loadCheckPoint(self, checkpoint_dir, agent, buffer):
    print('loading checkpoint')

    checkpoint = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'))
    args = checkpoint['args']
    agent.loadFromState(checkpoint['agent'])
    buffer.loadFromState(checkpoint['buffer_state'])

    self.num_eps = checkpoint['logger']['num_eps']
    self.num_steps = checkpoint['logger']['num_steps']
    self.num_training_steps = checkpoint['logger']['num_training_steps']
    self.training_eps_rewards = checkpoint['logger']['training_eps_rewards']
    self.num_eval_intervals = checkpoint['logger']['num_eval_intervals']
    self.eval_eps_rewards = checkpoint['logger']['eval_eps_rewards']
    self.eval_mean_values = checkpoint['logger']['eval_mean_values']
    self.eval_eps_lens = checkpoint['logger']['eval_eps_lens']
    self.loss = checkpoint['logger']['loss']

    torch.set_rng_state(checkpoint['torch_rng_state'])
    torch.cuda.set_rng_state(checkpoint['torch_cuda_rng_state'])
    np.random.set_state(checkpoint['np_rng_state'])

    if hasattr(agent, 'his'):
      agent.his = checkpoint['agent_his']

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
    if len(self.eval_eps_rewards) > 0:
      xs = np.arange(self.eval_freq, (len(self.eval_eps_rewards) + 1) * self.eval_freq, self.eval_freq)
      plt.plot(xs, self.eval_eps_rewards)
      plt.savefig(os.path.join(self.info_dir, 'eval_curve.pdf'))
      plt.close()

  def saveRewards(self):
    np.save(os.path.join(self.info_dir, 'rewards.npy'), self.training_eps_rewards)

  def saveLosses(self):
    np.save(os.path.join(self.info_dir, 'losses.npy'), self.loss)

  def saveEvalRewards(self):
    np.save(os.path.join(self.info_dir, 'eval_rewards.npy'), self.eval_eps_rewards)

  def exportData(self):
    super().exportData()
    self.saveLearningCurve()
    self.saveEvalCurve()
    self.saveLossCurve()
    self.saveRewards()
    self.saveEvalRewards()
    self.saveLosses()
import copy
import time
import gc

import numpy as np
import numpy.random as npr
import ray
import torch

from adn_agent import ADNAgent

@ray.remote
class Trainer(object):
  def __init__(self, initial_checkpoint, config):
    self.config = config
    if torch.cuda.is_available():
      self.device = torch.device('cuda')
    else:
      self.device = torch.device('cpu')

    self.agent = ADNAgent(self.config, self.device, training=True)

    self.agent.setWeights(copy.deepcopy(initial_checkpoint['weights']))
    if initial_checkpoint['optimizer_state'] is not None:
      self.agent.setOptimizerState(initial_checkpoint['optimizer_state'])

    self.training_step =  initial_checkpoint['training_step']

    npr.seed(self.config.seed)
    torch.manual_seed(self.config.seed)

  def continuousUpdateWeights(self, replay_buffer, shared_storage):
    while ray.get(shared_storage.getInfo.remote('num_eps')) <= 5:
      time.sleep(0.1)

    next_batch = replay_buffer.sample.remote(shared_storage)
    while self.training_step < self.config.training_steps and \
          not ray.get(shared_storage.getInfo.remote('terminate')):
      idx_batch, class_weight, batch = ray.get(next_batch)
      next_batch = replay_buffer.sample.remote(shared_storage)

      # Update target value model to use newest weights
      priorities, q_value_loss, state_value_loss, reward_loss, forward_loss, pred_obs = self.agent.updateWeights(batch, class_weight)

      replay_buffer.updatePriorities.remote(priorities, idx_batch)
      self.training_step += 1

      if self.training_step % self.config.decay_lr_interval == 0 and self.training_step > 0:
        self.agent.updateLR()

      if self.training_step % self.config.decay_action_sample_pen == 0 and self.training_step > 0:
        self.agent.decayActionSamplePen()

      if self.training_step % self.config.checkpoint_interval == 0:
        shared_storage.setInfo.remote(
          {
            'weights' : copy.deepcopy(self.agent.getWeights()),
            'optimizer_state' : copy.deepcopy(self.agent.getOptimizerState())
          }
        )
        replay_buffer.updateTargetNetwork.remote(shared_storage)
        if self.config.save_model:
          shared_storage.saveReplayBuffer.remote(replay_buffer.getBuffer.remote())
          shared_storage.saveCheckpoint.remote()

      sampled_idxs = ray.get(shared_storage.getInfo.remote('sampled_idxs'))
      idx, count = torch.unique(torch.Tensor(idx_batch)[:,1], return_counts=True)
      sampled_idxs[idx.long()] += count

      shared_storage.setInfo.remote(
        {
          'training_step' : self.training_step,
          'lr' : self.agent.getLR(),
          'q_value_loss' : q_value_loss,
          'state_value_loss' : state_value_loss,
          'reward_loss' : reward_loss,
          'forward_loss' : forward_loss,
          'class_weights' : class_weight,
          'sampled_idxs' : sampled_idxs,
          'training_pred' : pred_obs
        }
      )

      gc.collect()

      if self.config.training_delay:
        time.sleep(self.config.training_delay)

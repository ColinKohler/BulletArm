import copy
import os
import ray
import torch
import pickle

@ray.remote
class SharedStorage:
  def __init__(self, checkpoint, config):
    self.config = config
    self.current_checkpoint = copy.deepcopy(checkpoint)

  def saveCheckpoint(self, path=None):
    if not path:
      path = os.path.join(self.config.results_path, 'model.checkpoint')

    torch.save(self.current_checkpoint, path)

  def getCheckpoint(self):
    return copy.deepcopy(self.current_checkpoint)

  def saveReplayBuffer(self, replay_buffer):
    pickle.dump(
      {
        'buffer' : replay_buffer,
        'num_eps' : self.current_checkpoint['num_eps'],
        'num_steps' : self.current_checkpoint['num_steps'],
        'class_weights' : self.current_checkpoint['class_weights']
      },
      open(os.path.join(self.config.results_path, 'replay_buffer.pkl'), 'wb')
    )

  def logEpsReward(self, reward):
    self.current_checkpoint['eps_reward'].append(reward)

  def getInfo(self, keys):
    if isinstance(keys, str):
      return self.current_checkpoint[keys]
    elif isinstance(keys, list):
      return {key: self.current_checkpoint[key] for key in keys}
    else:
      raise TypeError

  def setInfo(self, keys, values=None):
    if isinstance(keys, str) and values is not None:
      self.current_checkpoint[keys] = values
    elif isinstance(keys, dict):
      self.current_checkpoint.update(keys)
    else:
      raise TypeError

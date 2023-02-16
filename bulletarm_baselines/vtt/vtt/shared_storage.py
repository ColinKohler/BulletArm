import copy
import os
import ray
import torch
import pickle
import numpy as np

@ray.remote
class SharedStorage(object):
  '''
  Remote ray worker class used to share data betweeen ray workers.

  Args:
    checkpoint (dict): Training checkpoint.
    config (dict): Task configuration.
  '''
  def __init__(self, checkpoint, config):
    self.config = config
    self.current_checkpoint = copy.deepcopy(checkpoint)

  def saveCheckpoint(self, path=None):
    '''
    Save the checkpoint to file.

    Args:
      path (str): The path to save the checkpoint to. Defaults to None.
        When set to None, defaults to results path in config.
    '''
    if not path:
      path = os.path.join(self.config.results_path, 'model.checkpoint')
    torch.save(self.current_checkpoint, path)

  def getCheckpoint(self):
    '''
    Get the current checkpoint.

    Returns:
      dict: Current checkpoint.
    '''
    return copy.deepcopy(self.current_checkpoint)

  def saveReplayBuffer(self, replay_buffer):
    '''
    Save the replay buffer to file.

    Args:
     replay_buffer (list): The replay buffer data.
    '''
    pickle.dump(
      {
        'buffer' : replay_buffer,
        'num_eps' : self.current_checkpoint['num_eps'],
        'num_steps' : self.current_checkpoint['num_steps']
      },
      open(os.path.join(self.config.results_path, 'replay_buffer.pkl'), 'wb')
    )

  def getInfo(self, keys):
    '''
    Get data from the current checkpoint for the desired keys.

    Args:
      keys (str | list[str]): Keys to get data from.

    Returns:
      dict: The key-value pairs desired.
    '''
    if isinstance(keys, str):
      return self.current_checkpoint[keys]
    elif isinstance(keys, list):
      return {key: self.current_checkpoint[key] for key in keys}
    else:
      raise TypeError

  def setInfo(self, keys, values=None):
    '''
    Update the current checkpoint to the new key-value pairs.

    Args:
      keys (str | dict): Keys to update.
      values (list[str]): Values to update.
    '''
    if isinstance(keys, str) and values is not None:
      self.current_checkpoint[keys] = values
    elif isinstance(keys, dict):
      self.current_checkpoint.update(keys)
    else:
      raise TypeError

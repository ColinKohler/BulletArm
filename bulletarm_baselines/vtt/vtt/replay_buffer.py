import ray
import copy
import torch
import numpy as np
import numpy.random as npr

from bulletarm_baselines.vtt.vtt import torch_utils
from functools import partial

@ray.remote
class ReplayBuffer(object):
  '''

  '''
  def __init__(self, initial_checkpoint, initial_buffer, config):
    self.config = config
    self.config.batch_size = 32
    if self.config.seed:
      npr.seed(self.config.seed)

    self.buffer = copy.deepcopy(initial_buffer)
    self.num_eps = initial_checkpoint['num_eps']
    self.num_steps = initial_checkpoint['num_steps']
    self.total_samples = sum([len(eps_history.vision_history) for eps_history in self.buffer.values()])

  def getBuffer(self):
    '''
    Get the replay buffer.

    Returns:
      list[EpisodeHistory] : The replay buffer
    '''
    return self.buffer

  def add(self, eps_history, shared_storage=None):
    '''
    Add a new episode to the replay buffer. If the episode already has priorities
    those are used, otherwise we calculate them in the standard TD error fashion:
    td_error = |V(s,a) - (V(s', a') * R(s,a) ** gamma)| + eps
    priority = td_error ** alpha

    Args:
      eps_history (EpisodeHistory): The episode to add to the buffer.
      shared_storage (ray.Worker): Shared storage worker. Defaults to None.
    '''
    if eps_history.priorities is None:
      priorities = list()
      for i, value in enumerate(eps_history.value_history):
        if (i + 1) < len(eps_history.value_history):
          priority = np.abs(value - (eps_history.reward_history[i] + self.config.discount * eps_history.value_history[i+1])) + self.config.per_eps
        else:
          priority = np.abs(value - eps_history.reward_history[i]) + self.config.per_eps
        priority += 1 if eps_history.is_expert else 0
        priorities.append(priority ** self.config.per_alpha)

      eps_history.priorities = np.array(priorities, dtype=np.float32)
      eps_history.eps_priority = np.max(eps_history.priorities)

    # Add to buffer
    self.buffer[self.num_eps] = copy.deepcopy(eps_history)
    self.num_eps += 1
    self.num_steps += len(eps_history.vision_history)
    self.total_samples += len(eps_history.vision_history)

    # Delete the oldest episode if the buffer is full
    if self.config.replay_buffer_size < len(self.buffer):
      del_id = self.num_eps - len(self.buffer)
      self.total_samples -= len(self.buffer[del_id].vision_history)
      del self.buffer[del_id]

    if shared_storage:
      shared_storage.setInfo.remote('num_eps', self.num_eps)
      shared_storage.setInfo.remote('num_steps', self.num_steps)

  def sample(self, shared_storage):
    '''
    Sample a batch from the replay buffer.

    Args:
      shared_storage (ray.Worker): Shared storage worker.

    Returns:
      (list[int], list[numpy.array], list[numpy.array], list[double], list[double]) : (Index, Observation, Action, Reward, Weight)
    '''
    (index_batch,
     vision_batch,
     force_batch,
     proprio_batch,
     next_vision_batch,
     next_force_batch,
     next_proprio_batch,
     action_batch,
     reward_batch,
     done_batch,
     is_expert_batch,
     weight_batch
    ) = [list() for _ in range(12)]

    for _ in range(self.config.batch_size):
      eps_id, eps_history, eps_prob = self.sampleEps(uniform=False)
      eps_step, step_prob = self.sampleStep(eps_history, uniform=False)

      force = eps_history.force_history[eps_step].reshape(self.config.force_history, self.config.force_dim)
      force_ = eps_history.force_history[eps_step+1].reshape(self.config.force_history, self.config.force_dim)

      proprio = eps_history.proprio_history[eps_step].reshape(1, self.config.proprio_dim)
      proprio_ = eps_history.proprio_history[eps_step+1].reshape(1, self.config.proprio_dim)

      vision, vision_, = self.crop(
        eps_history.vision_history[eps_step],
        eps_history.vision_history[eps_step+1],
      )
      action = eps_history.action_history[eps_step+1]

      index_batch.append([eps_id, eps_step])
      vision_batch.append(vision)
      force_batch.append(force)
      proprio_batch.append(proprio)
      next_vision_batch.append(vision_)
      next_force_batch.append(force_)
      next_proprio_batch.append(proprio_)
      action_batch.append(action)
      reward_batch.append(eps_history.reward_history[eps_step+1])
      done_batch.append(eps_history.done_history[eps_step+1])
      is_expert_batch.append(eps_history.is_expert)

      training_step = ray.get(shared_storage.getInfo.remote('training_step'))
      weight_batch.append((1 / (self.total_samples * eps_prob * step_prob)) ** self.config.getPerBeta(training_step))

    vision_batch = torch.tensor(np.stack(vision_batch)).float()
    force_batch = torch.tensor(np.stack(force_batch)).float()
    proprio_batch = torch.tensor(np.stack(proprio_batch)).float()
    next_vision_batch = torch.tensor(np.stack(next_vision_batch)).float()
    next_force_batch = torch.tensor(np.stack(next_force_batch)).float()
    next_proprio_batch = torch.tensor(np.stack(next_proprio_batch)).float()
    action_batch = torch.tensor(np.stack(action_batch)).float()
    reward_batch = torch.tensor(reward_batch).float()
    done_batch = torch.tensor(done_batch).int()
    # non_final_mask_batch = (done_batch ^ 1).float()
    non_final_mask_batch = (done_batch).float()
    is_expert_batch = torch.tensor(is_expert_batch).long()
    weight_batch = torch.tensor(weight_batch).float()

    return (
      index_batch,
      (
        (vision_batch, force_batch, proprio_batch),
        (next_vision_batch, next_force_batch, next_proprio_batch),
        action_batch,
        reward_batch,
        non_final_mask_batch,
        is_expert_batch,
        weight_batch
      )
    )

  def sampleEps(self, uniform=False):
    '''
    Sample a episode from the buffer using the priorities

    Returns:
      (int, EpisodeHistory, double) : (episode ID, episode, episode probability)
    '''
    if uniform:
      eps_idx = npr.choice(len(self.buffer))
      eps_prob = 1.0
    else:
      eps_probs = np.array([eps_history.eps_priority for eps_history in self.buffer.values()], dtype=np.float32)
      eps_probs /= np.sum(eps_probs)

      eps_idx = npr.choice(len(self.buffer), p=eps_probs)
      eps_prob = eps_probs[eps_idx]

    eps_id = self.num_eps - len(self.buffer) + eps_idx
    return eps_id, self.buffer[eps_id], eps_prob

  def sampleStep(self, eps_history, uniform=False):
    '''
    Sample a step from the given episode using the step priorities

    Args:
      eps_history (EpisodeHistory): The episode to sample a step from

    Returns:
      (int, double) : (step index, step probability)
    '''
    if uniform:
      step_idx = npr.choice(len(eps_history.priorities[:-1]))
      step_prob = 1.0
    else:
      step_probs = eps_history.priorities[:-1] / sum(eps_history.priorities[:-1])
      step_idx = npr.choice(len(step_probs), p=step_probs)
      step_prob = step_probs[step_idx]

    return step_idx, step_prob

  def augmentTransitionSO2(self, vision, vision_):
    ''''''
    vision_aug, vision_aug_, transform_params = torch_utils.perturb(
      vision.copy(),
      vision_.copy(),
      set_theta_zero=True
    )

    vision = vision_aug.reshape(*vision.shape)
    vision_ = vision_aug_.reshape(*vision_.shape)

    return vision, vision_

  def crop(self, vision, vision_):
    s = vision.shape[-1]

    crop_max = s - self.config.vision_size + 1
    w1 = npr.randint(0, crop_max)
    w2 = npr.randint(0, crop_max)
    vision = vision[:, w1:w1+self.config.vision_size, w2:w2+self.config.vision_size]
    vision_ = vision_[:, w1:w1+self.config.vision_size, w2:w2+self.config.vision_size]

    return vision, vision_

  def updatePriorities(self, td_errors, idx_info):
    '''
    Update the priorities for each sample in the batch.

    Args:
      td_errors (numpy.array): The TD error for each sample in the batch
      idx_info (numpy.array): The episode and step for each sample in the batch
    '''
    for i in range(len(idx_info)):
      eps_id, eps_step = idx_info[i]

      if next(iter(self.buffer)) <= eps_id:
        td_error = td_errors[i]

        self.buffer[eps_id].priorities[eps_step] = (td_error + (1 if self.buffer[eps_id].is_expert else 0) + self.config.per_eps) ** self.config.per_alpha
        self.buffer[eps_id].eps_priority = np.max(self.buffer[eps_id].priorities)

  def resetPriorities(self):
    '''
    Uniformly reset the priorities for all samples in the buffer.
    '''
    for eps_history in self.buffer.values():
      eps_history.eps_priority = 1.0
      eps_history.priorities = np.array([1.0] * len(eps_history.priorities))

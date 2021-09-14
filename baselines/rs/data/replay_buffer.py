import gc
import copy
import ray

import numpy as np
import numpy.random as npr
import torch

from data import data_utils
from adn_agent import ADNAgent
import utils

@ray.remote
class ReplayBuffer():
  def __init__(self, initial_checkpoint, initial_buffer, config, sampler_workers):
    '''
    Create replay buffer.

    Args:
    '''
    self.config = config
    npr.seed(self.config.seed)

    self.sampler_workers = sampler_workers
    self.buffer = copy.deepcopy(initial_buffer)
    self.num_eps = initial_checkpoint['num_eps']
    self.num_steps = initial_checkpoint['num_steps']
    self.total_samples = sum([len(eps_history.value_history) for eps_history in self.buffer.values()])
    self.class_counts = np.zeros(self.config.num_depth_classes)

  def __len__(self):
    return len(self.buffer)

  def getBuffer(self):
    return self.buffer

  def add(self, eps_history, shared_storage=None):
    '''
    Add experience to replay buffer.
    '''
    if eps_history.priorities is not None:
      eps_history.priorities = np.copy(eps_history.priorities)
    else:
      priorities = list()
      per_eps = self.config.expert_per_eps if eps_history.expert_traj else self.config.per_eps
      for i, value in enumerate(eps_history.value_history):
        if (i + 1) < len(eps_history.value_history):
          priority = np.abs(value - (eps_history.reward_history[i] + self.config.discount * eps_history.value_history[i+1])) + per_eps
        else:
          priority = np.abs(value - eps_history.reward_history[i]) + per_eps

        priorities.append(priority ** self.config.per_alpha)

        if i > 0:
          obs_prob = data_utils.convertDepthToOneHot(eps_history.obs_history[i][2], self.config.num_depth_classes)
          action = torch.Tensor(eps_history.action_history[i]).view(1, -1)
          action = torch.Tensor([action[0,2], action[0,1], action[0,3]]).view(1, -1)
          deictic_obs = utils.getDeicticActions(obs_prob, action)
          deictic_label = data_utils.convertProbToLabel(deictic_obs, self.config.num_depth_classes)
          values, counts = torch.unique(deictic_label, return_counts=True)
          for v, c in zip(values, counts):
            self.class_counts[v] += c

      eps_history.priorities = np.array(priorities, dtype=np.float32)
      eps_history.eps_priority = np.max(eps_history.priorities)

    self.buffer[self.num_eps] = copy.deepcopy(eps_history)
    self.num_eps += 1
    self.num_steps += len(eps_history.value_history)
    self.total_samples += len(eps_history.value_history)

    if self.config.replay_buffer_size < len(self.buffer):
      del_id = self.num_eps - len(self.buffer)
      self.total_samples -= len(self.buffer[del_id].value_history)
      del self.buffer[del_id]

    if shared_storage:
      shared_storage.setInfo.remote('num_eps', self.num_eps)
      shared_storage.setInfo.remote('num_steps', self.num_steps)

  def sample(self, shared_storage):
    (index_batch,
     state_batch,
     hand_obs_batch,
     obs_batch,
     action_batch,
     reward_batch,
     state_value_batch,
     q_value_batch,
     weight_batch
    ) = [list() for _ in range(9)]

    for _ in range(int(self.config.batch_size / len(self.sampler_workers))):
      samples = list()
      for sampler_worker in self.sampler_workers:
        eps_id, eps_history, eps_prob = self.sampleEps(force_uniform=False)
        eps_step, step_prob = self.sampleStep(eps_history, force_uniform=False)

        idx = [eps_id, eps_step]

        training_step = ray.get(shared_storage.getInfo.remote('training_step'))
        beta = self.config.getPerBeta(training_step)
        weight = (1 / (self.total_samples * eps_prob * step_prob)) ** beta

        samples.append(sampler_worker.makeTarget.remote(eps_history, eps_step, idx, weight))

      while len(samples):
        done_id, samples = ray.wait(samples)
        if done_id:
          s = done_id[0]
          idx, weight, state_values, q_values, rewards, state, hand_obs, obs, actions = ray.get(s)

          index_batch.append(idx)
          state_batch.append(state)
          hand_obs_batch.append(torch.stack(hand_obs))
          obs_batch.append(torch.stack(obs))
          action_batch.append(actions)
          state_value_batch.append(state_values)
          q_value_batch.append(q_values)
          reward_batch.append(rewards)
          weight_batch.append(weight)

    weight_batch = np.array(weight_batch, dtype=np.float32) / max(weight_batch)

    state_batch = torch.tensor(state_batch).long()
    hand_obs_batch = torch.stack(hand_obs_batch).float()
    obs_batch = torch.stack(obs_batch).float()
    action_batch = torch.tensor(action_batch).float()
    state_value_batch = torch.tensor(state_value_batch).float()
    q_value_batch = torch.tensor(q_value_batch).float()
    reward_batch = torch.tensor(reward_batch).float()
    weight_batch = torch.tensor(weight_batch.copy()).float()

    class_weight = torch.ones(self.config.num_depth_classes)
    for i, c in enumerate(self.class_counts):
      class_weight[i] = np.log(0.05 * np.sum(self.class_counts) / c) if c != 0 else 1
    class_weight = class_weight.clamp_(1)

    batch = (state_batch,
             hand_obs_batch,
             obs_batch,
             action_batch,
             state_value_batch,
             q_value_batch,
             reward_batch,
             weight_batch)

    return index_batch, class_weight, batch

  def sampleEps(self, force_uniform=False):
    eps_prob = None
    if not force_uniform:
      eps_probs = np.array([eps_history.eps_priority for eps_history in list(self.buffer.values())], dtype=np.float32)
      eps_probs /= np.sum(eps_probs)
      try:
        eps_idx = npr.choice(len(eps_probs), p=eps_probs)
      except:
        breakpoint()
      eps_prob = eps_probs[eps_idx]
    else:
      eps_idx = npr.choice(len(self.buffer))
    eps_id = self.num_eps - len(self.buffer) + eps_idx

    return eps_id, self.buffer[eps_id], eps_prob

  def sampleStep(self, eps_history, force_uniform=False):
    step_prob = None
    if not force_uniform:
      step_probs = (eps_history.priorities[:-1]) / sum(eps_history.priorities[:-1])
      step_idx = npr.choice(len(step_probs), p=step_probs)
      step_prob = step_probs[step_idx]
    else:
      step_idx = npr.choice(len(eps_history.value_history))

    return step_idx, step_prob

  def updateEpsHistory(self, eps_id, eps_history):
    if next(iter(self.buffer)) <= eps_id:
      # Note: Avoid read only array when loading replay buffer from disk
      eps_history.priorities = np.copy(eps_history.priorities)
      self.buffer[eps_id] = eps_history

  def updatePriorities(self, td_errors, idx_info):
    for i in range(len(idx_info)):
      eps_id, eps_step = idx_info[i]

      if next(iter(self.buffer)) <= eps_id:
        td_error = td_errors[i,:]
        start_idx = eps_step
        end_idx = min(eps_step + len(td_error), len(self.buffer[eps_id].priorities))
        per_eps = self.config.expert_per_eps if self.buffer[eps_id].expert_traj else self.config.per_eps
        self.buffer[eps_id].priorities[start_idx:end_idx] = (td_error[:end_idx-start_idx] + per_eps) ** self.config.per_alpha

        self.buffer[eps_id].eps_priority = np.max(self.buffer[eps_id].priorities)

  def resetPriorities(self):
    for eps_history in self.buffer.values():
      eps_history.eps_priority = 1.0
      eps_history.priorities = np.array([1.0] * len(eps_history.priorities))

  def updateTargetNetwork(self, shared_storage):
    [sampler_worker.updateTargetNetwork.remote(shared_storage) for sampler_worker in self.sampler_workers]

@ray.remote
class Sampler():
  def __init__(self, initial_checkpoint, config, seed):
    npr.seed(seed)
    self.config = config

    if torch.cuda.is_available():
      self.device = torch.device('cuda')
    else:
      self.device = torch.device('cpu')

    self.agent = ADNAgent(self.config, self.device)
    self.agent.setWeights(initial_checkpoint['weights'])

  def computeTargetValue(self, eps_history, idx):
    if idx + 1 < len(eps_history.obs_history):
      q_map, q_maps, sampled_actions, pixel_action, pred_obs, values = self.agent.selectAction(
        eps_history.obs_history[idx],
        normalize_obs=False,
        return_all_states=True
      )

      state_value = eps_history.reward_history[idx] + self.config.discount * values[np.argmax(values)]
      q_value = eps_history.reward_history[idx] + self.config.discount * torch.max(q_maps).item()
    else:
      num_sampled_actions = 1
      state_value = eps_history.reward_history[idx]
      q_value = eps_history.reward_history[idx]

    return state_value, q_value

  def makeTarget(self, eps_history, step_idx, idx, weight):
    target_state_values, target_q_values, target_rewards, state, hand_obs, obs, actions = [list() for _ in range(7)]
    for current_idx in range(step_idx, step_idx + self.config.num_unroll_steps + 1):
      if current_idx < len(eps_history.value_history):
        state_value, q_value = self.computeTargetValue(eps_history, current_idx)
        target_state_values.append(state_value)
        target_q_values.append(q_value)
        target_rewards.append(eps_history.reward_history[current_idx])
        actions.append(eps_history.action_history[current_idx])

        state.append(eps_history.obs_history[current_idx][0])
        hand_obs.append(data_utils.convertDepthToOneHot(eps_history.obs_history[current_idx][1],
                                                        self.config.num_depth_classes).squeeze())
        obs.append(data_utils.convertDepthToOneHot(eps_history.obs_history[current_idx][2],
                                                   self.config.num_depth_classes).squeeze())
      else:
        target_state_values.append(target_state_values[-1])
        target_q_values.append(target_q_values[-1])
        target_rewards.append(target_rewards[-1])
        actions.append([0,0,0,0])
        state.append(state[-1])
        hand_obs.append(hand_obs[-1])
        obs.append(obs[-1])

    gc.collect()

    return idx, weight, target_state_values, target_q_values, target_rewards, state, hand_obs, obs, actions

  def updateTargetNetwork(self, shared_storage):
    training_step = ray.get(shared_storage.getInfo.remote('training_step'))
    if training_step % self.config.decay_action_sample_pen == 0 and training_step > 0:
      self.agent.decayActionSamplePen()
    self.agent.setWeights(ray.get(shared_storage.getInfo.remote('weights')))

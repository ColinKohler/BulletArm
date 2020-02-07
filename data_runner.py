import numpy as np
import torch
from multiprocessing import Process, Pipe
import os
import git
import helping_hands_rl_envs

def worker(remote, parent_remote, env_fn, planner_fn):
  '''
  Worker function which interacts with the environment over remote

  Args:
    - remote: Worjer remote connection
    - parent_remote: Data EnvRunner remote connection
    - env_fn: Function which creates a deictic environment
  '''
  parent_remote.close()

  env = env_fn()
  planner = planner_fn(env)

  try:
    while True:
      cmd, data = remote.recv()
      if cmd == 'step':
        obs, reward, done = env.step(data)
        valid = env.isSimValid()
        remote.send((obs, reward, done, valid))
      elif cmd == 'reset':
        obs = env.reset()
        remote.send(obs)
      elif cmd == 'close':
        remote.close()
        break
      elif cmd == 'get_next_action':
        remote.send(planner.getNextAction())
      elif cmd == 'did_block_fall':
        remote.send(env.didBlockFall())
      elif cmd == 'get_value':
        remote.send(planner.getValue())
      elif cmd == 'get_steps_left':
        remote.send(planner.getStepsLeft())
      else:
        raise NotImplementerError
  except KeyboardInterrupt:
    print('EnvRunner worker: caught keyboard interrupt')

class DataRunner(object):
  '''
  Data environment runner which runs mulitpl environemnts in parallel in subprocesses
  and communicates with them via pipe

  Args:
    - envs: List of DeiciticEnvs
  '''
  def __init__(self, env_fns, planner_fns):
    self.waiting = False
    self.closed = False

    num_envs = len(env_fns)
    self.remotes, self.worker_remotes = zip(*[Pipe() for _ in range(num_envs)])
    self.processes = [Process(target=worker, args=(worker_remote, remote, env_fn, planner_fn))
                      for (worker_remote, remote, env_fn, planner_fn) in zip(self.worker_remotes, self.remotes, env_fns, planner_fns)]
    self.num_processes = len(self.processes)

    for process in self.processes:
      process.daemon = True
      process.start()
    for remote in self.worker_remotes:
      remote.close()

  def step(self, actions):
    '''
    Step the environments synchronously.

    Args:
      - actions: PyTorch variable of environment actions
    '''
    self.stepAsync(actions)
    return self.stepWait()

  def stepAsync(self, actions):
    '''
    Step each environment in a async fashion

    Args:
      - actions: PyTorch variable of environment actions
    '''
    actions = actions.squeeze(1).numpy()
    for remote, action in zip(self.remotes, actions):
      remote.send(('step', action))
    self.waiting = True

  def stepWait(self):
    '''
    Wait until each environment has completed its next step

    Returns: (obs, rewards, dones)
      - obs: Torch vector of observations
      - rewards: Torch vector of rewards
      - dones: Numpy vector of 0/1 flags indicating if episode is done
    '''
    results = [remote.recv() for remote in self.remotes]
    self.waiting = False

    obs, rewards, dones, valids = zip(*results)
    states, hand_obs, depths = zip(*obs)

    states = torch.from_numpy(np.stack(states).astype(float)).float()
    hand_obs = torch.from_numpy(np.stack(hand_obs)).float()
    depths = torch.from_numpy(np.stack(depths)).float()
    rewards = torch.from_numpy(np.stack(rewards)).float()
    if len(rewards.shape) == 1:
      rewards = rewards.unsqueeze(1)
    dones = torch.from_numpy(np.stack(dones).astype(np.float32)).float()
    valids = torch.from_numpy(np.stack(valids).astype(np.float32)).float()

    return states, hand_obs, depths, rewards, dones, valids

  def reset(self):
    '''
    Reset each environment

    Returns: Torch vector of observations
    '''
    for remote in self.remotes:
      remote.send(('reset', None))

    obs = [remote.recv() for remote in self.remotes]
    states, hand_obs, depths = zip(*obs)

    states = torch.from_numpy(np.stack(states).astype(float)).float()
    hand_obs = torch.from_numpy(np.stack(hand_obs)).float()
    depths = torch.from_numpy(np.stack(depths)).float()

    return states, hand_obs, depths

  def reset_envs(self, env_nums):
    for env_num in env_nums:
      self.remotes[env_num].send(('reset', None))

    obs = [self.remotes[env_num].recv() for env_num in env_nums]
    states, hand_obs, depths = zip(*obs)

    states = torch.from_numpy(np.stack(states).astype(float)).float()
    hand_obs = torch.from_numpy(np.stack(hand_obs)).float()
    depths = torch.from_numpy(np.stack(depths)).float()

    return states, hand_obs, depths

  def close(self):
    '''
    Close all worker processes
    '''
    self.closed = True
    if self.waiting:
      [remote.recv() for remote in self.remotes]
    [remote.send(('close', None)) for remote in self.remotes]
    [process.join() for process in self.processes]

  def getNextAction(self):
    for remote in self.remotes:
      remote.send(('get_next_action', None))
    action = [remote.recv() for remote in self.remotes]
    action = torch.from_numpy(np.stack(action)).float()
    return action

  def getStepsLeft(self):
    for remote in self.remotes:
      remote.send(('get_steps_left', None))
    steps_left = [remote.recv() for remote in self.remotes]
    steps_left = torch.from_numpy(np.stack(steps_left)).float()
    return steps_left

  def getValue(self):
    for remote in self.remotes:
      remote.send(('get_value', None))
    values = [remote.recv() for remote in self.remotes]
    values = torch.from_numpy(np.stack(values)).float()
    return values

  def didBlockFall(self):
    for remote in self.remotes:
      remote.send(('did_block_fall', None))
    did_block_fall = [remote.recv() for remote in self.remotes]
    did_block_fall = torch.from_numpy(np.stack(did_block_fall)).float()
    return did_block_fall

  def setPosCandidate(self, pos_candidate):
    for remote in self.remotes:
      remote.send(('set_pos_candidate', pos_candidate))

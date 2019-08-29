import numpy as np
import torch
from multiprocessing import Process, Pipe
import os

def worker(remote, parent_remote, env_fn):
  '''
  Worker function which interacts with the environment over remote

  Args:
    - remote: Worjer remote connection
    - parent_remote: EnvRunner remote connection
    - env_fn: Function which creates a deictic environment
  '''
  parent_remote.close()
  env = env_fn()
  try:
    while True:
      cmd, data = remote.recv()
      if cmd == 'step':
        obs, reward, done = env.step(data)
        remote.send((obs, reward, done))
      elif cmd == 'step_auto_reset':
        obs, reward, done = env.step(data)
        if done: obs = env.reset()
        remote.send((obs, reward, done))
      elif cmd == 'reset':
        obs = env.reset()
        remote.send(obs)
      elif cmd == 'save':
        env.saveState()
      elif cmd == 'restore':
        env.restoreState()
      elif cmd == 'close':
        remote.close()
        break
      elif cmd == 'get_spaces':
        remote.send((env.obs_shape, env.action_space, env.action_shape))
      elif cmd == 'get_obj_position':
        remote.send(env.getObjectPosition())
      elif cmd == 'get_plan':
        remote.send(env.getPlan())
      elif cmd == 'set_pos_candidate':
        env.setPosCandidate(data)
      elif cmd == 'save_to_file':
        path = data
        env.saveEnvToFile(path)
      elif cmd == 'load_from_file':
        path = data
        env.loadEnvFromFile(path)
      else:
        raise NotImplementerError
  except KeyboardInterrupt:
    print('EnvRunner worker: caught keyboard interrupt')
  # finally:

class EnvRunner(object):
  '''
  Environment runner which runs mulitpl environemnts in parallel in subprocesses
  and communicates with them via pipe

  Args:
    - envs: List of DeiciticEnvs
  '''
  def __init__(self, env_fns):
    self.waiting = False
    self.closed = False

    num_envs = len(env_fns)
    self.remotes, self.worker_remotes = zip(*[Pipe() for _ in range(num_envs)])
    self.processes = [Process(target=worker, args=(worker_remote, remote, env_fn))
                      for (worker_remote, remote, env_fn) in zip(self.worker_remotes, self.remotes, env_fns)]
    self.num_processes = len(self.processes)

    for process in self.processes:
      process.daemon = True
      process.start()
    for remote in self.worker_remotes:
      remote.close()

    self.remotes[0].send(('get_spaces', None))
    self.obs_shape, self.action_space, self.action_shape = self.remotes[0].recv()

  def step(self, actions, auto_reset=True):
    '''
    Step the environments synchronously.

    Args:
      - actions: PyTorch variable of environment actions
    '''
    self._stepAsync(actions, auto_reset)
    return self._stepWait()

  def _stepAsync(self, actions, auto_reset=True):
    '''
    Step each environment in a async fashion

    Args:
      - actions: PyTorch variable of environment actions
    '''
    actions = actions.squeeze(1).numpy()
    for remote, action in zip(self.remotes, actions):
      if auto_reset:
        remote.send(('step_auto_reset', action))
      else:
        remote.send(('step', action))
    self.waiting = True

  def _stepWait(self):
    '''
    Wait until each environment has completed its next step

    Returns: (obs, rewards, dones)
      - obs: Torch vector of observations
      - rewards: Torch vector of rewards
      - dones: Numpy vector of 0/1 flags indicating if episode is done
    '''
    results = [remote.recv() for remote in self.remotes]
    self.waiting = False

    obs, rewards, dones = zip(*results)
    states, depths = zip(*obs)

    states = torch.from_numpy(np.stack(states).astype(float)).float()
    depths = torch.from_numpy(np.stack(depths)).float()
    rewards = torch.from_numpy(np.stack(rewards)).float()
    if len(rewards.shape) == 1:
      rewards = rewards.unsqueeze(1)
    dones = torch.from_numpy(np.stack(dones).astype(np.float32)).float()

    return states, depths, rewards, dones

  def reset(self):
    '''
    Reset each environment

    Returns: Torch vector of observations
    '''
    for remote in self.remotes:
      remote.send(('reset', None))

    obs = [remote.recv() for remote in self.remotes]
    states, depths = zip(*obs)

    states = torch.from_numpy(np.stack(states).astype(float)).float()
    depths = torch.from_numpy(np.stack(depths)).float()

    return states, depths

  def close(self):
    '''
    Close all worker processes
    '''
    self.closed = True
    if self.waiting:
      [remote.recv() for remote in self.remotes]
    [remote.send(('close', None)) for remote in self.remotes]
    [process.join() for process in self.processes]

  def save(self):
    for remote in self.remotes:
      remote.send(('save', None))

  def restore(self):
    for remote in self.remotes:
      remote.send(('restore', None))

  def saveToFile(self, path):
    for i, remote in enumerate(self.remotes):
      p = os.path.join(path, str(i))
      if not os.path.exists(p):
        os.makedirs(p)
      remote.send(('save_to_file', os.path.join(path, str(i))))

  def loadFromFile(self, path):
    for i, remote in enumerate(self.remotes):
      remote.send(('load_from_file', os.path.join(path, str(i))))

  def getObjPosition(self):
    for remote in self.remotes:
      remote.send(('get_obj_position', None))

    position = [remote.recv() for remote in self.remotes]
    return position

  def getPlan(self):
    for remote in self.remotes:
      remote.send(('get_plan', None))
    plan = [remote.recv() for remote in self.remotes]
    plan = torch.from_numpy(np.stack(plan)).float()
    return plan

  def setPosCandidate(self, pos_candidate):
    for remote in self.remotes:
      remote.send(('set_pos_candidate', pos_candidate))

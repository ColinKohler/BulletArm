import numpy as np
import torch
from multiprocessing import Process, Pipe

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
  env.connectToVrep()
  try:
    while True:
      cmd, data = remote.recv()
      if cmd == 'step':
        obs, reward, done = env.step(data)
        if done: obs = env.reset()
        remote.send((obs, reward, done))
      elif cmd == 'reset':
        obs = env.reset()
        remote.send(obs)
      elif cmd == 'close':
        remote.close()
        break
      elif cmd == 'get_spaces':
        remote.send((env.obs_shape, env.action_space, env.action_shape))
      else:
        raise NotImplementerError
  except KeyboardInterrupt:
    print('EnvRunner worker: caught keyboard interrupt')
  finally:
    env.disconnectToVrep()

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

  def step(self, actions):
    '''
    Step the environments synchronously.

    Args:
      - actions: PyTorch variable of environment actions
    '''
    self._stepAsync(actions)
    return self._stepWait()

  def _stepAsync(self, actions):
    '''
    Step each environment in a async fashion

    Args:
      - actions: PyTorch variable of environment actions
    '''
    actions = actions.squeeze(1).numpy()
    for remote, action in zip(self.remotes, actions):
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
    rewards = torch.from_numpy(np.stack(rewards)).unsqueeze(dim=1).float()
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

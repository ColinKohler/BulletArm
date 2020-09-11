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
    - remote: Worker remote connection
    - parent_remote: RL EnvRunner remote connection
    - env_fn: Function which creates a deictic environment
  '''
  parent_remote.close()
  env = env_fn()
  planner = planner_fn(env)

  try:
    while True:
      cmd, data = remote.recv()
      if cmd == 'step':
        res = env.step(data)
        remote.send(res)
      elif cmd == 'step_auto_reset':
        res = env.step(data)
        done = res[2]
        if done:
          # get observation after reset (res index 0), the rest stays the same
          res = (env.reset(), *res[1:])
        remote.send(res)
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
      elif cmd == 'get_obs':
        action = data
        if action is None:
          action = env.last_action
        obs = env._getObservation(action)
        remote.send(obs)
      elif cmd == 'get_spaces':
        remote.send((env.obs_shape, env.action_space, env.action_shape))
      elif cmd == 'get_obj_positions':
        remote.send(env.getObjectPositions())
      elif cmd == 'get_obj_poses':
        remote.send(env.getObjectPoses())
      elif cmd == 'set_pos_candidate':
        env.setPosCandidate(data)
      elif cmd == 'get_next_action':
        remote.send(planner.getNextAction())
      elif cmd == 'did_block_fall':
        remote.send(env.didBlockFall())
      elif cmd == 'get_value':
        remote.send(planner.getValue())
      elif cmd == 'get_step_left':
        remote.send(planner.getStepLeft())
      elif cmd == 'get_empty_in_hand':
        remote.send(env.getEmptyInHand())
      elif cmd == 'save_to_file':
        path = data
        env.saveEnvToFile(path)
      elif cmd == 'load_from_file':
        try:
          path = data
          env.loadEnvFromFile(path)
        except Exception as e:
          print('EnvRunner worker load failed: {}'.format(e))
          remote.send(False)
        else:
          remote.send(True)
      else:
        raise NotImplementerError
  except KeyboardInterrupt:
    print('EnvRunner worker: caught keyboard interrupt')

class RLRunner(object):
  '''
  RL environment runner which runs mulitpl environemnts in parallel in subprocesses
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

    self.remotes[0].send(('get_spaces', None))
    self.obs_shape, self.action_space, self.action_shape = self.remotes[0].recv()

  def step(self, actions, auto_reset=True):
    '''
    Step the environments synchronously.

    Args:
      - actions: PyTorch variable of environment actions
    '''
    self.stepAsync(actions, auto_reset)
    return self.stepWait()

  def stepAsync(self, actions, auto_reset=True):
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

    res = tuple(zip(*results))

    if len(res) == 3:
      return_metadata = False
      metadata = None
      obs, rewards, dones = res
    else:
      return_metadata = True
      obs, rewards, dones, metadata = res

    states, hand_obs, depths = zip(*obs)

    states = torch.from_numpy(np.stack(states).astype(float)).float()
    hand_obs = torch.from_numpy(np.stack(hand_obs)).float()
    depths = torch.from_numpy(np.stack(depths)).float()
    rewards = torch.from_numpy(np.stack(rewards)).float()
    if len(rewards.shape) == 1:
      rewards = rewards.unsqueeze(1)
    dones = torch.from_numpy(np.stack(dones).astype(np.float32)).float()

    if return_metadata:
      return states, hand_obs, depths, rewards, dones, metadata
    else:
      return states, hand_obs, depths, rewards, dones

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
    return np.array([remote.recv() for remote in self.remotes]).all()

  def getObjPositions(self):
    for remote in self.remotes:
      remote.send(('get_obj_positions', None))

    positions = [remote.recv() for remote in self.remotes]
    return np.array(positions)

  def getObjPoses(self):
    for remote in self.remotes:
      remote.send(('get_obj_poses', None))

    poses = [remote.recv() for remote in self.remotes]
    return np.array(poses)

  def getNextAction(self):
    for remote in self.remotes:
      remote.send(('get_next_action', None))
    action = [remote.recv() for remote in self.remotes]
    action = torch.from_numpy(np.stack(action)).float()
    return action

  def getValue(self):
    for remote in self.remotes:
      remote.send(('get_value', None))
    values = [remote.recv() for remote in self.remotes]
    values = torch.from_numpy(np.stack(values)).float()
    return values

  def getStepLeft(self):
    for remote in self.remotes:
      remote.send(('get_step_left', None))
    values = [remote.recv() for remote in self.remotes]
    values = torch.from_numpy(np.stack(values)).float()
    return values

  def getObs(self, action=None):
    for remote in self.remotes:
      remote.send(('get_obs', action))

    obs = [remote.recv() for remote in self.remotes]
    states, hand_obs, depths = zip(*obs)

    states = torch.from_numpy(np.stack(states).astype(float)).float()
    hand_obs = torch.from_numpy(np.stack(hand_obs)).float()
    depths = torch.from_numpy(np.stack(depths)).float()

    return states, hand_obs, depths

  def didBlockFall(self):
    for remote in self.remotes:
      remote.send(('did_block_fall', None))
    did_block_fall = [remote.recv() for remote in self.remotes]
    did_block_fall = torch.from_numpy(np.stack(did_block_fall)).float()
    return did_block_fall

  def setPosCandidate(self, pos_candidate):
    for remote in self.remotes:
      remote.send(('set_pos_candidate', pos_candidate))

  def getEmptyInHand(self):
    for remote in self.remotes:
      remote.send(('get_empty_in_hand', None))
    hand_obs = [remote.recv() for remote in self.remotes]
    hand_obs = torch.from_numpy(np.stack(hand_obs)).float()
    return hand_obs

  @staticmethod
  def getEnvGitHash():
    repo = git.Repo(helping_hands_rl_envs.__path__[0])
    return repo.head.object.hexsha

'''
.. moduleauthor: Colin Kohler <github.com/ColinKohler>
'''

import numpy as np
from multiprocessing import Process, Pipe
import os
import git
import bulletarm
from tqdm import tqdm
import copy
import time

def worker(remote, parent_remote, env_fn, planner_fn=None):
  '''
  Worker function which interacts with the environment over the remove connection

  Args:
    remote (multiprocessing.Connection): Worker remote connection
    parent_remote (multiprocessing.Connection): MultiRunner remote connection
    env_fn (function): Creates a environment
    planner_fn (function): Creates the planner for the environment
  '''
  parent_remote.close()

  env = env_fn()
  if planner_fn:
    planner = planner_fn(env)
  else:
    planner = None

  try:
    while True:
      cmd, data = remote.recv()
      if cmd == 'step':
        res = env.step(data)
        remote.send(res)
      elif cmd == 'simulate':
        res = env.simulate(data)
        remote.send(res)
      elif cmd == 'can_simulate':
        remote.send(env.canSimulate())
      elif cmd == 'reset_sim':
        env.resetSimPose()
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
      elif cmd == 'get_spaces':
        remote.send((env.obs_shape, env.action_space, env.action_shape))
      elif cmd == 'get_empty_in_hand':
        remote.send(env.getEmptyInHand())
      elif cmd == 'get_env_id':
        remote.send(env.active_env_id)
      elif cmd == 'get_next_action':
        if planner:
          remote.send(planner.getNextAction())
        else:
          raise ValueError('Attempting to use a planner which was not initialized.')
      elif cmd == 'get_num_obj':
        remote.send(env.num_obj)
      elif cmd == 'save':
        env.saveState()
      elif cmd == 'restore':
        env.restoreState()
      elif cmd == 'save_to_file':
        path = data
        env.saveEnvToFile(path)
      elif cmd == 'load_from_file':
        try:
          path = data
          env.loadEnvFromFile(path)
        except Exception as e:
          print('MultiRunner worker load failed: {}'.format(e))
          remote.send(False)
        else:
          remote.send(True)
      elif cmd == 'close':
        remote.close()
        break
      else:
        raise NotImplementerError
  except KeyboardInterrupt:
    print('MultiRunner worker: caught keyboard interrupt')

class MultiRunner(object):
  '''
  Runner which runs mulitple environemnts in parallel in subprocesses and communicates with them via pipe.

  Args:
    env_fns (list[function]): Env creation functions
    planner_fns (list[function]): Planner creation functions
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

  def step(self, actions, auto_reset=False):
    '''
    Step the environments synchronously.

    Args:
      actions (numpy.array): Actions to take in each environment
      auto_reset (bool): Reset environments automatically after an episode ends

    Returns:
      (numpy.array, numpy.array, numpy.array): (observations, rewards, done flags)
    '''
    self.stepAsync(actions, auto_reset)
    return self.stepWait()

  def simulate(self, actions):
    for remote, action in zip(self.remotes, actions):
      remote.send(('simulate', action))

    obs = [remote.recv() for remote in self.remotes]
    states, hand_obs, obs = zip(*obs)

    states = np.stack(states).astype(float)
    hand_obs = np.stack(hand_obs)
    obs = np.stack(obs)
    rewards = np.zeros_like(states).astype(np.float32)
    dones = np.zeros_like(states).astype(np.float32)

    return (states, hand_obs, obs), rewards, dones

  def canSimulate(self):
    for remote in self.remotes:
      remote.send(('can_simulate', None))
    flag = [remote.recv() for remote in self.remotes]
    flag = np.stack(flag)
    return flag

  def resetSimPose(self):
    for remote in self.remotes:
      remote.send(('reset_sim', None))

  def stepAsync(self, actions, auto_reset=False):
    '''
    Step each environment in a async fashion.

    Args:
      actions (numpy.array): Actions to take in each environment
      auto_reset (bool): Reset environments automatically after an episode ends
    '''
    for remote, action in zip(self.remotes, actions):
      if auto_reset:
        remote.send(('step_auto_reset', action))
      else:
        remote.send(('step', action))
    self.waiting = True

  def stepWait(self):
    '''
    Wait until each environment has completed its next step.

    Returns:
      (numpy.array, numpy.array, numpy.array): (observations, rewards, done flags)
    '''
    results = [remote.recv() for remote in self.remotes]
    self.waiting = False

    res = tuple(zip(*results))

    if len(res) == 3:
      metadata = None
      obs, rewards, dones = res
    else:
      obs, rewards, dones, metadata = res

    states, hand_obs, obs = zip(*obs)

    states = np.stack(states).astype(float)
    hand_obs = np.stack(hand_obs)
    obs = np.stack(obs)
    rewards = np.stack(rewards)
    dones = np.stack(dones).astype(np.float32)

    if metadata:
      return (states, hand_obs, obs), rewards, dones, metadata
    else:
      return (states, hand_obs, obs), rewards, dones

  def reset(self):
    '''
    Reset each environment.

    Returns:
      numpy.array: Observations
    '''
    for remote in self.remotes:
      remote.send(('reset', None))

    obs = [remote.recv() for remote in self.remotes]
    states, hand_obs, obs = zip(*obs)

    states = np.stack(states).astype(float)
    hand_obs = np.stack(hand_obs)
    obs = np.stack(obs)

    return (states, hand_obs, obs)

  def reset_envs(self, env_nums):
    '''
    Resets the specified environments.

    Args:
      env_nums (list[int]): The environments to be reset

    Returns:
      numpy.array: Observations
    '''

    for env_num in env_nums:
      self.remotes[env_num].send(('reset', None))

    obs = [self.remotes[env_num].recv() for env_num in env_nums]
    states, hand_obs, obs = zip(*obs)

    states = np.stack(states).astype(float)
    hand_obs = np.stack(hand_obs)
    obs = np.stack(obs)

    return (states, hand_obs, obs)

  def close(self):
    '''
    Close all worker processes.
    '''
    self.closed = True
    if self.waiting:
      [remote.recv() for remote in self.remotes]
    [remote.send(('close', None)) for remote in self.remotes]
    [process.join() for process in self.processes]

  def save(self):
    '''
    Locally saves the current state of the environments.
    '''
    for remote in self.remotes:
      remote.send(('save', None))

  def restore(self):
    '''
    Restores the locally saved state of the environments.
    '''
    for remote in self.remotes:
      remote.send(('restore', None))

  def saveToFile(self, path):
    '''
    Saves the current state of the environments to file.

    Args:
      path (str): The path to save the enviornment states to
    '''
    for i, remote in enumerate(self.remotes):
      p = os.path.join(path, str(i))
      if not os.path.exists(p):
        os.makedirs(p)
      remote.send(('save_to_file', os.path.join(path, str(i))))

  def loadFromFile(self, path):
    '''
    Loads the state of the environments from a file.

    Args:
      path (str): The path to the environment states to load

    Returns:
      bool: Flag indicating if the loading succeeded for all environments
    '''
    for i, remote in enumerate(self.remotes):
      remote.send(('load_from_file', os.path.join(path, str(i))))
    return np.array([remote.recv() for remote in self.remotes]).all()

  def getNextAction(self):
    '''
    Get the next action from the planner for each environment.

    Returns:
      numpy.array: Actions
    '''
    for remote in self.remotes:
      remote.send(('get_next_action', None))
    action = [remote.recv() for remote in self.remotes]
    action = np.stack(action)
    return action

  def getEmptyInHand(self):
    '''

    '''
    for remote in self.remotes:
      remote.send(('get_empty_in_hand', None))
    hand_obs = [remote.recv() for remote in self.remotes]
    hand_obs = np.stack(hand_obs)
    return hand_obs

  @staticmethod
  def getEnvGitHash():
    '''

    '''
    repo = git.Repo(bulletarm.__path__[0])
    return repo.head.object.hexsha

  def getNumObj(self):
    '''
    Get the number of objects in the environment
    Returns: int: number of objects
    '''
    self.remotes[0].send(('get_num_obj', None))
    num_obj = self.remotes[0].recv()
    return num_obj

  def gatherDeconstructTransitions(self, planner_episode):
    '''
    Gather deconstruction transitions and reverse them for construction

    Args:
      - planner_episode: The number of expert episodes to gather

    Returns: list of transitions. Each transition is in the form of
    ((state, in_hand, obs), action, reward, done, (next_state, next_in_hand, next_obs))
    '''
    num_processes = self.num_processes
    num_objects = self.getNumObj()
    def states_valid(states_list):
      if len(states_list) < 2:
        return False
      for i in range(1, len(states_list)):
        if states_list[i] != 1 - states_list[i-1]:
          return False
      return True

    def rewards_valid(reward_list):
      if reward_list[0] != 1:
        return False
      for i in range(1, len(reward_list)):
        if reward_list[i] != 0:
          return False
      return True

    def getCurrentObs(in_hand, obs):
      obss = []
      for i, o in enumerate(obs):
        obss.append((o.squeeze(), in_hand[i].squeeze()))
      return obss

    states, in_hands, obs = self.reset()
    total = 0
    s = 0
    step_times = []
    steps = [0 for i in range(num_processes)]
    local_state = [[] for i in range(num_processes)]
    local_obs = [[] for i in range(num_processes)]
    local_action = [[] for i in range(num_processes)]
    local_reward = [[] for i in range(num_processes)]

    pbar = tqdm(total=planner_episode)

    transitions = []
    while total < planner_episode:
      plan_actions = self.getNextAction()
      actions_star = np.concatenate((plan_actions, np.expand_dims(states, 1)), axis=1)
      t0 = time.time()
      (states_, in_hands_, obs_), rewards, dones = self.step(actions_star, auto_reset=False)
      dones[states + states_ != 1] = 1
      t = time.time()-t0
      step_times.append(t)

      buffer_obs = getCurrentObs(in_hands_, obs)
      for i in range(num_processes):
        local_state[i].append(states[i])
        local_obs[i].append(buffer_obs[i])
        local_action[i].append(plan_actions[i])
        local_reward[i].append(rewards[i])

      steps = list(map(lambda x: x + 1, steps))

      done_idxes = np.nonzero(dones)[0]
      if done_idxes.shape[0] != 0:
        empty_in_hands = self.getEmptyInHand()

        buffer_obs_ = getCurrentObs(empty_in_hands, copy.deepcopy(obs_))
        reset_states_, reset_in_hands_, reset_obs_ = self.reset_envs(done_idxes)
        for i, idx in enumerate(done_idxes):
          local_obs[idx].append(buffer_obs_[idx])
          local_state[idx].append(copy.deepcopy(states_[idx]))
          if (num_objects-2)*2 <= steps[idx] <= num_objects*2 and states_valid(local_state[idx]) and rewards_valid(local_reward[idx]):
            s += 1
            for j in range(len(local_reward[idx])):
              obs = local_obs[idx][j+1]
              next_obs = local_obs[idx][j]
              transitions.append(((local_state[idx][j+1], obs[1], obs[0]), # (state, in_hand, obs)
                                  local_action[idx][j], # action
                                  local_reward[idx][j], # reward
                                  j == 0, # done
                                  (local_state[idx][j], next_obs[1], next_obs[0]))) # (next_state, next_in_hand, next_obs)

          states_[idx] = reset_states_[i]
          obs_[idx] = reset_obs_[i]

          total += 1
          steps[idx] = 0
          local_state[idx] = []
          local_obs[idx] = []
          local_action[idx] = []
          local_reward[idx] = []

      pbar.set_description(
        '{}/{}, SR: {:.3f}, step time: {:.2f}; avg step time: {:.2f}'
          .format(s, total, float(s)/total if total !=0 else 0, t, np.mean(step_times))
      )
      pbar.update(done_idxes.shape[0])

      states = copy.copy(states_)
      obs = copy.copy(obs_)
    pbar.close()
    return transitions

class SingleRunner(object):
  '''
  RL environment runner which runs a single environment.

  Args:
    env (BaseEnv): Environment
    planner (BasePlanner): Planner
  '''
  def __init__(self, env, planner=None):
    self.env = env
    self.planner = planner

  def step(self, action, auto_reset=True):
    '''
    Step the environment.

    Args:
      action (numpy.array): Action to take in the environment
      auto_reset (bool): Reset the environment after an episode ends

    Returns:
      (numpy.array, numpy.array, numpy.array): (observations, rewards, done flags)
    '''
    results = self.env.step(action)

    if len(results) == 3:
      metadata = None
      obs, rewards, dones = results
    else:
      obs, rewards, dones, metadata = results
    states, hand_obs, obs = obs

    if metadata:
      return (states, hand_obs, obs), rewards, dones, metadata
    else:
      return (states, hand_obs, obs), rewards, dones

  def reset(self):
    '''
    Reset the environment.

    Returns:
      numpy.array: Observation
    '''
    return self.env.reset()

  def save(self):
    '''
    Locally saves the current state of the environment.
    '''
    self.env.save()

  def restore(self):
    '''
    Restores the locally saved state of the environment.
    '''
    self.env.restore()

  def saveToFile(self, path):
    '''
    Saves the current state of the environment to file.

    Args:
      path (str): The path to save the enviornment state to
    '''
    self.env.saveToFile(path)

  def loadFromFile(self, path):
    '''
    Loads the state of the environment from a file.

    Args:
      path (str): The path to the environment state to load

    Returns:
      bool: Flag indicating if the loading succeeded
    '''
    return self.env.loadFromFile(path)

  def getNextAction(self):
    '''

    '''
    if self.planner:
      return self.planner.getNextAction()
    else:
      raise ValueError('Attempting to use a planner which was not initialized.')

  @staticmethod
  def getEnvGitHash():
    repo = git.Repo(bulletarm.__path__[0])
    return repo.head.object.hexsha

  def close(self):
    return

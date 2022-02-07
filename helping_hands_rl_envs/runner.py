'''
.. moduleauthor: Colin Kohler <github.com/ColinKohler>
'''

import numpy as np
from multiprocessing import Process, Pipe
import os
import git
import helping_hands_rl_envs

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
      elif cmd == 'get_obs':
        obs = env._getObservation(env.last_action)
        remote.send(obs)
      elif cmd == 'get_spaces':
        remote.send((env.obs_shape, env.action_space, env.action_shape))
      elif cmd == 'get_object_positions':
        remote.send(env.getObjectPositions())
      elif cmd == 'get_object_poses':
        remote.send(env.getObjectPoses())
      elif cmd == 'set_pos_candidate':
        env.setPosCandidate(data)
      elif cmd == 'did_block_fall':
        remote.send(env.didBlockFall())
      elif cmd == 'are_objects_in_workspace':
        remote.send(env.areObjectsInWorkspace())
      elif cmd == 'is_sim_valid':
        remote.send(env.isSimValid())
      elif cmd == 'get_value':
        remote.send(planner.getValue())
      elif cmd == 'get_step_left':
        remote.send(planner.getStepLeft())
      elif cmd == 'get_active_env_id':
        remote.send(env.active_env_id)
      elif cmd == 'get_empty_in_hand':
        remote.send(env.getEmptyInHand())
      elif cmd == 'get_env_id':
        remote.send(env.active_env_id)
      elif cmd == 'get_next_action':
        if planner:
          remote.send(planner.getNextAction())
        else:
          raise ValueError('Attempting to use a planner which was not initialized.')
      elif cmd == 'get_random_action':
        if planner:
          remote.send(planner.getRandomAction())
        else:
          raise ValueError('Attempting to use a planner which was not initialized.')
      elif cmd == 'get_value':
        if planner:
          remote.send(planner.getValue())
        else:
          raise ValueError('Attempting to use a planner which was not initialized.')
      elif cmd == 'get_steps_left':
        if planner:
          remote.send(planner.getStepsLeft())
        else:
          raise ValueError('Attempting to use a planner which was not initialized.')
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

  def getActiveEnvId(self):
    '''
    '''
    for remote in self.remotes:
      remote.send(('get_active_env_id', None))
    active_env_id = [remote.recv() for remote in self.remotes]
    active_env_id = np.stack(active_env_id)

    return active_env_id

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

  def getObjectPositions(self):
    '''

    '''
    for remote in self.remotes:
      remote.send(('get_object_positions', None))

    positions = [remote.recv() for remote in self.remotes]
    return np.array(positions)

  def getObjectPoses(self):
    '''

    '''
    for remote in self.remotes:
      remote.send(('get_object_poses', None))

    poses = [remote.recv() for remote in self.remotes]
    return np.array(poses)

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

  def getRandomAction(self):
    '''
    Get a random action for each environment.

    Returns:
      numpy.array: Actions
    '''
    for remote in self.remotes:
      remote.send(('get_random_action', None))
    action = [remote.recv() for remote in self.remotes]
    action = np.stack(action)
    return action

  def getValue(self):
    '''

    '''
    for remote in self.remotes:
      remote.send(('get_value', None))
    values = [remote.recv() for remote in self.remotes]
    values = np.stack(values)
    return values

  def getStepsLeft(self):
    '''

    '''
    for remote in self.remotes:
      remote.send(('get_steps_left', None))
    values = [remote.recv() for remote in self.remotes]
    values = np.stack(values)
    return values

  def getObs(self):
    '''
    Get the current observation for the environments.

    Returns:
      (numpy.array, numpy.array, numpy.array): (hand state, in-hand observation, workspace observation)
    '''
    for remote in self.remotes:
      remote.send(('get_obs'))

    obs = [remote.recv() for remote in self.remotes]
    states, hand_obs, obs = zip(*obs)

    states = np.stack(states).astype(float)
    hand_obs = np.stack(hand_obs)
    obs = np.stack(obs)

    return states, hand_obs, obs

  def areObjectsInWorkspace(self):
    '''

    '''
    for remote in self.remotes:
      remote.send(('are_objects_in_workspace', None))
    in_workspace = [remote.recv() for remote in self.remotes]
    in_workspace = np.stack(in_workspace)
    return in_workspace

  def isSimValid(self):
    '''

    '''
    for remote in self.remotes:
      remote.send(('is_sim_valid', None))
    valid = [remote.recv() for remote in self.remotes]
    valid = np.stack(valid)
    return valid

  def didBlockFall(self):
    '''

    '''
    for remote in self.remotes:
      remote.send(('did_block_fall', None))
    did_block_fall = [remote.recv() for remote in self.remotes]
    did_block_fall = np.stack(did_block_fall)
    return did_block_fall

  def setPosCandidate(self, pos_candidate):
    '''

    '''
    for remote in self.remotes:
      remote.send(('set_pos_candidate', pos_candidate))

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
    repo = git.Repo(helping_hands_rl_envs.__path__[0])
    return repo.head.object.hexsha

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

  def getObjectPositions(self):
    '''

    '''
    return self.env.getObjectPositions()

  def getObjectPoses(self):
    '''

    '''
    return self.env.getObjectPoses()

  def getNextAction(self):
    '''

    '''
    if self.planner:
      return self.planner.getNextAction()
    else:
      raise ValueError('Attempting to use a planner which was not initialized.')

  def getRandomAction(self):
    '''

    '''
    if self.planner:
      return self.planner.getRandomAction()
    else:
      raise ValueError('Attempting to use a planner which was not initialized.')

  def getValue(self):
    '''

    '''
    if self.planner:
      return self.planner.getValue()
    else:
      raise ValueError('Attempting to use a planner which was not initialized.')

  def getStepsLeft(self):
    '''

    '''
    if self.planner:
      return self.planner.getStepsLeft()
    else:
      raise ValueError('Attempting to use a planner which was not initialized.')

  def getActiveEnvId(self):
    '''

    '''
    return self.env.active_env_id

  def areObjectsInWorkspace(self):
    '''

    '''
    if self.planner:
      return self.planner.areObjectsInWorkspace()
    else:
      raise ValueError('Attempting to use a planner which was not initialized.')

  def isSimValid(self):
    '''

    '''
    return self.env.isSimValid()


  def getObs(self, action=None):
    '''

    '''
    return self.env._getObservation(action if action else self.env.last_action)

  def didBlockFall(self):
    '''

    '''
    return self.env.didBlockFall()

  def setPosCandidate(self, pos_candidate):
    '''

    '''
    self.env.setPosCandidate(pos_candidate)

  def getEmptyInHand(self):
    '''

    '''
    return self.env.getEmptyInHand()

  @staticmethod
  def getEnvGitHash():
    repo = git.Repo(helping_hands_rl_envs.__path__[0])
    return repo.head.object.hexsha

  def close(self):
    return

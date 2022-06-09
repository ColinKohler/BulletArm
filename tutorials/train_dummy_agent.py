import numpy as np
from bulletarm import env_factory

class DummyAgent(object):
  def __init__(self):
    pass

  def getAction(self, obs):
    return None

  def train(self, batch):
    pass

def sample(data):
  return None

def trainDummyAgent():
  # Define some hyperparameters
  num_envs = 2
  num_episodes_to_generate = 10
  num_training_steps = 100

  # Create two envs to be run in parallel to speed up training
  env_config = {'render': False}
  envs = env_factory.createEnvs(num_envs, 'block_stacking', env_config)

  agent = DummyAgent()
  num_episodes = 0
  data = list()

  # Generate data
  while num_episodes < num_episodes_to_generate:
    obs = envs.reset()
    actions = envs.getNextAction()
    obs_, rewards, dones = envs.step(actions)

    # Add transitions to data: (s, a, s', r, d)
    for i in range(num_envs):
      data.append((obs[i], actions[i], obs_[i], rewards[i], dones[i]))

    # Reset envs which have completed episodes
    done_idxs = np.nonzero(dones)[0]
    if len(done_idxs) != 0:
      new_obs_ = self.envs.reset_envs(done_idxs)
  envs.close()

  # Train agent
  for _ in range(num_training_steps):
    batch = sample(data)
    agent.train(batch)

if __name__ == '__main__':
  trainDummyAgent()

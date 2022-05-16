import gc
import time
import numpy as np
import numpy.random as npr
import ray
import torch
import functools

from data.episode_history import EpisodeHistory
from adn_agent import ADNAgent
from data import data_utils
from data import constants
import utils

from bulletarm import env_factory

@ray.remote
class DataGenerator(object):
  def __init__(self, initial_checkpoint, config, seed):
    self.seed = seed
    self.config = config
    self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    env_type, env_config, _ = data_utils.getEnvConfig(self.config.env_type, self.config.use_rot)
    self.env = env_factory.createEnvs(0,  env_type, env_config)

    npr.seed(self.seed)
    torch.manual_seed(self.seed)

    self.agent = ADNAgent(self.config, self.device)
    self.agent.setWeights(initial_checkpoint['weights'])

  def continuousDataGen(self, shared_storage, replay_buffer, test_mode=False):
    while (
      ray.get(shared_storage.getInfo.remote('training_step')) < self.config.training_steps and \
      not ray.get(shared_storage.getInfo.remote('terminate'))
    ):
      # NOTE: This might be a inificent but we can't check the training step really due to async
      self.agent.setWeights(ray.get(shared_storage.getInfo.remote('weights')))

      training_step = ray.get(shared_storage.getInfo.remote('training_step'))
      if training_step % self.config.decay_action_sample_pen == 0 and training_step > 0:
        self.agent.decayActionSamplePen()

      if not test_mode:
        eps_history = self.generateEpisode(test_mode)

        replay_buffer.add.remote(eps_history, shared_storage)
        shared_storage.logEpsReward.remote(eps_history.reward_history[-1])
        gc.collect()
      else:
        eps_history = self.generateEpisode(test_mode)
        past_100_rewards = ray.get(shared_storage.getInfo.remote('past_100_rewards'))
        past_100_rewards.append(eps_history.reward_history[-1])

        shared_storage.setInfo.remote(
          {
            'eps_len' : len(eps_history.action_history),
            'total_reward' : sum(eps_history.reward_history),
            'past_100_rewards': past_100_rewards,
            'mean_value' : np.mean([value for value in eps_history.value_history]),
            'eps_obs' : [eps_history.obs_history, eps_history.pred_obs_history],
            'eps_values' : [np.round(value, 2) for value in eps_history.value_history],
            'eps_sampled_actions' : eps_history.sampled_action_history,
            'eps_q_maps' : eps_history.q_map_history
          }
        )
        gc.collect()

      if not test_mode and self.config.data_gen_delay:
        time.sleep(self.config.gen_delay)
      if not test_mode and self.config.train_data_ratio:
        while(
            ray.get(shared_storage.getInfo.remote('training_step'))
            / max(1, ray.get(shared_storage.getInfo.remote('num_steps')))
            < self.config.train_data_ratio):
          time.sleep(0.5)

  def generateEpisode(self, test_mode):
    eps_history = EpisodeHistory()

    obs = self.env.reset()
    obs_rb = (obs[0], self.agent.preprocessDepth(obs[1]), self.agent.preprocessDepth(obs[2]))
    with torch.no_grad():
      value, reward = self.agent.state_value_model(
        torch.Tensor(obs_rb[2]).to(self.device),
        torch.Tensor(obs_rb[1]).to(self.device)
      )
    eps_history.value_history.append(value.item())
    eps_history.action_history.append([0,0,0,0])
    eps_history.obs_history.append(obs_rb)
    eps_history.reward_history.append(0.0)

    done = False
    while not done:
      state = int(obs[0])
      q_map, q_maps, sampled_actions, pixel_action, pred_obs, value = self.agent.selectAction(obs)
      pixel_action = pixel_action.tolist()
      action = utils.getWorkspaceAction(pixel_action, constants.WORKSPACE, constants.OBS_RESOLUTION, self.agent.rotations)

      obs, reward, done = self.env.step(action.cpu().numpy(), auto_reset=False)

      if np.max(obs[2]) > self.config.max_height:
        done = True
        continue

      obs_rb = [obs[0], self.agent.preprocessDepth(obs[1]), self.agent.preprocessDepth(obs[2])]
      eps_history.value_history.append(value)
      eps_history.sampled_action_history.append(sampled_actions)
      eps_history.action_history.append(pixel_action)
      eps_history.obs_history.append(obs_rb)
      if test_mode:
        if pred_obs is not None:
          eps_history.pred_obs_history.append(
            data_utils.convertProbToDepth(pred_obs, self.config.num_depth_classes).squeeze().cpu().numpy()
          )
        else:
          eps_history.pred_obs_history.append(obs_rb[2].squeeze())
        eps_history.q_map_history.append(q_map.cpu().numpy().squeeze())
      eps_history.reward_history.append(reward)

    eps_history.q_map_history.append(np.zeros((self.config.obs_size, self.config.obs_size)))
    eps_history.sampled_action_history.append(None)

    return eps_history

@ray.remote
class ExpertDataGenerator(object):
  def __init__(self, initial_checkpoint, config, seed):
    self.seed = seed
    self.config = config
    self.device = torch.device('cpu')

    env_type, env_config, planner_config = data_utils.getEnvConfig(self.config.expert_env,
                                                                   self.config.use_rot)
    self.env = env_factory.createEnvs(0,  env_type, env_config, planner_config=planner_config)

    self.preprocessDepth = functools.partial(data_utils.preprocessDepth,
                                             min=0.,
                                             max=self.config.max_height,
                                             num_classes=self.config.num_depth_classes,
                                             round=2,
                                             noise=False)

    npr.seed(self.seed)
    torch.manual_seed(self.seed)

  def continuousDataGen(self, shared_storage, replay_buffer):
    while (
      ray.get(shared_storage.getInfo.remote('training_step')) < self.config.training_steps and \
      not ray.get(shared_storage.getInfo.remote('terminate'))
    ):
      if 'deconstruct' in self.config.expert_env:
        valid_eps, eps_history = self.generateEpisodeWithDeconstruct()
      else:
        valid_eps, eps_history = self.generateEpisode()

      if len(eps_history.obs_history) == 1 or not valid_eps:
        continue
      else:
        replay_buffer.add.remote(eps_history, shared_storage)

    ray.actor.exit_actor()

  def generateEpisodeWithDeconstruct(self):
    eps_history = EpisodeHistory(expert_traj=True)
    actions = list()
    obs = self.env.reset()

    # Deconstruct structure while saving the actions reversing the action primative
    done = False
    while not done:
      action = self.env.getNextAction()

      primative = np.abs(action[0] - 1)
      actions.append([primative, action[1], action[2], action[3]])

      obs, reward, done = self.env.step(action, auto_reset=False)

    obs_rb = [obs[0], self.preprocessDepth(obs[1]), self.preprocessDepth(obs[2])]
    valid_eps = self.env.isSimValid()

    eps_history.value_history.append(0.0)
    eps_history.action_history.append([0,0,0,0])
    eps_history.obs_history.append(obs_rb)
    eps_history.reward_history.append(0)

    for i, action in enumerate(actions[::-1]):
      rotations = torch.from_numpy(np.linspace(0, np.pi, self.config.num_rots, endpoint=False))
      rot_idx = np.abs(rotations - action[3]).argmin()
      action[-1] = rotations[rot_idx]

      pixel_action = utils.getPixelAction(action, self.config.workspace, self.config.obs_resolution, self.config.obs_size).tolist()
      pixel_action = [pixel_action[0], pixel_action[2], pixel_action[1], rot_idx]

      obs, reward, done = self.env.step(action, auto_reset=False)
      obs_rb = [obs[0], self.preprocessDepth(obs[1]), self.preprocessDepth(obs[2])]

      if not self.env.isSimValid():
        break

      if self.env.didBlockFall():
        reward = 0
      else:
        reward = 1 if i == len(actions) - 1 else 0

      eps_history.value_history.append(0.0)
      eps_history.action_history.append(pixel_action)
      eps_history.obs_history.append(obs_rb)
      eps_history.reward_history.append(reward)

    return valid_eps, eps_history

  def generateEpisode(self):
    eps_history = EpisodeHistory(expert_traj=True)

    obs = self.env.reset()
    obs_rb = [obs[0], self.preprocessDepth(obs[1]), self.preprocessDepth(obs[2])]

    eps_history.value_history.append(0.0)
    eps_history.action_history.append([0,0,0,0])
    eps_history.obs_history.append(obs_rb)
    eps_history.reward_history.append(0)

    done = False

    while not done:
      action = self.env.getNextAction()
      rotations = torch.from_numpy(np.linspace(0, np.pi, self.config.num_rots, endpoint=False))
      rot_idx = np.abs(rotations - action[3]).argmin()
      action[-1] = rotations[rot_idx]

      obs, reward, done = self.env.step(action, auto_reset=False)

      pixel_action = utils.getPixelAction(action, self.config.workspace, self.config.obs_resolution, self.config.obs_size).tolist()
      pixel_action = [pixel_action[0], pixel_action[2], pixel_action[1], rot_idx]

      obs_rb = [obs[0], self.preprocessDepth(obs[1]), self.preprocessDepth(obs[2])]

      eps_history.value_history.append(0.0)
      eps_history.action_history.append(pixel_action)
      eps_history.obs_history.append(obs_rb)
      eps_history.reward_history.append(reward)

    return True, eps_history

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from adn_agent import ADNAgent
from data import data_utils
from data import constants
import utils

from bulletarm import env_factory

def initAgent(task_config, use_cuda, checkpoint=None):
  device = torch.device('cuda') if use_cuda else torch.device('cpu')
  agent = ADNAgent(task_config, device)

  if checkpoint:
    agent.loadWeights(checkpoint)

  return agent

def plotEpisode(real_obs, model_obs, q_maps, sampled_actions, pixel_actions, values, rewards, save_path=None):
  fig, ax = plt.subplots(figsize=(15,10), nrows=4, ncols=len(model_obs))
  ax[0][0].set_ylabel('Obs', rotation=0, size='large', labelpad=30)
  ax[1][0].set_ylabel('Pred Obs', rotation=0, size='large', labelpad=30)
  for i in range(len(model_obs)):
    ax[0][i].set_title('T: {} | V: {:.2f} | R: {:.0f}'.format(i+1, values[i], np.abs(rewards[i])))
    ax[0][i].imshow(real_obs[i][2].squeeze(), cmap='gray')
    if i > 0: ax[0][i].scatter([int(pixel_actions[i-1][2])], [int(pixel_actions[i-1][1])], c='r', s=1)
    ax[1][i].imshow(model_obs[i].squeeze(), cmap='gray')
    if i > 0: ax[1][i].scatter([int(pixel_actions[i-1][2])], [int(pixel_actions[i-1][1])], c='r', s=1)
    ax[2][i].imshow(real_obs[i][1].squeeze(), cmap='gray')
    ax[2][i].scatter([32], [32], c='r', s=1)
    ax[3][i].imshow(q_maps[i].squeeze())
    if i < len(model_obs)-1:
      if sampled_actions[i] is not None:
        ax[3][i].scatter(sampled_actions[i][:,1], sampled_actions[i][:,0], c='r', s=1)
      ax[3][i].scatter([int(pixel_actions[i][2])], [int(pixel_actions[i][1])], c='white', s=1)
  fig.tight_layout()

  if save_path:
    fig.savefig(save_path)
    plt.close()
  else:
    plt.show()

def runEpisode(env, task_config, agent):
  obs = env.reset()
  real_obs = [[obs[0], obs[1], obs[2]]]
  model_obs = [obs[2]]
  value, reward = agent.getStateValue(obs)
  values = [value.item()]
  rewards = [reward.item()]
  q_maps = list()
  actions = list()
  pixel_actions = list()

  done = False
  while (not done):
    q_map, _, sampled_actions, pixel_action, pred_obs, value = agent.selectAction(obs)
    action = utils.getWorkspaceAction(pixel_action, constants.WORKSPACE, constants.OBS_RESOLUTION, agent.rotations)

    obs, r, done = env.step(action.cpu().numpy(), auto_reset=False)

    real_obs.append([obs[0], obs[1], obs[2]])
    if pred_obs is not None:
      model_obs.append(
        data_utils.convertProbToDepth(pred_obs, task_config.num_depth_classes).squeeze().cpu().numpy()
      )
    else:
      model_obs.append(obs[2])
    values.append(value)
    rewards.append(r)
    q_maps.append(q_map.cpu().numpy())
    pixel_actions.append(pixel_action)
    actions.append(sampled_actions)

  q_maps.append(q_map.cpu().numpy())

  return real_obs, model_obs, q_maps, actions, pixel_actions, values, rewards

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('task', type=str,
      help='Task to traing on')
  parser.add_argument('checkpoint', type=str,
      help='Path to the checkpoint to load')
  parser.add_argument('--num_eps', type=int, default=1,
      help='Number of episodes to benchmark on')
  parser.add_argument('--save_plots', default=False, action='store_true',
      help='Save plots from test results instead of displaying')
  parser.add_argument('--cuda', default=False, action='store_true',
      help='Use GPU for test.')
  args = parser.parse_args()

  task_config = data_utils.getTaskConfig(args.task, 1)
  env_type, env_config, _ = data_utils.getEnvConfig(task_config.env_type, task_config.use_rot, render=False)
  planner_config = data_utils.getPlannerConfig([0., 0.], [0., 0.], 0., 'play')
  env = env_factory.createEnvs(0,  env_type, env_config, planner_config=planner_config)
  agent = initAgent(task_config, args.cuda, args.checkpoint)

  if args.save_plots:
    utils.removeFiles(constants.BENCHMARK_RESULT_PATH + '/success/')
    utils.removeFiles(constants.BENCHMARK_RESULT_PATH + '/failure/')

  num_success = 0
  planning_failures = 0
  planning_time = list()
  planning_steps = list()
  pbar = tqdm.tqdm(total=args.num_eps)
  pbar.set_description('Success Rate: 0')
  for eps in range(args.num_eps):
    real_obs, model_obs, q_maps, sampled_actions, pixel_actions, values, rewards = runEpisode(env, task_config, agent)

    num_success += rewards[-1]
    if args.save_plots:
      if rewards[-1] == 1:
        fig_path = constants.BENCHMARK_RESULT_PATH + '/success/{}.png'.format(eps)
      else:
        fig_path = constants.BENCHMARK_RESULT_PATH + '/failure/{}.png'.format(eps)
    else:
      fig_path = None
    plotEpisode(real_obs, model_obs, q_maps, sampled_actions, pixel_actions, values, rewards, save_path=fig_path)

    pbar.update(1)
    pbar.set_description('Success Rate: {:.3f}'.format(int((num_success / (eps+1)) * 100)))

  pbar.close()

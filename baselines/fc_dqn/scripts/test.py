import os
import sys
import time
import copy
import math
import collections
from tqdm import tqdm

import torch

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

from scripts.create_agent import createAgent
from scripts.main import addPerlinNoiseToInHand, addPerlinNoiseToObs
from utils.parameters import *
from storage.buffer import QLearningBufferExpert, QLearningBuffer
from helping_hands_rl_envs import env_factory
from utils.logger import Logger
from utils.schedules import LinearSchedule
from utils.torch_utils import rand_perlin_2d
from utils.env_wrapper import EnvWrapper


ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done step_left expert')


def test():
    plt.style.use('default')
    envs = EnvWrapper(num_processes, simulator, env, env_config, planner_config)
    agent = createAgent()

    agent.loadModel(load_model_pre)
    agent.eval()
    states, in_hands, obs = envs.reset()
    obs = obs.permute(0, 3, 1, 2)
    in_hands = in_hands.permute(0, 3, 1, 2)
    test_episode = 1000
    total = 0
    s = 0
    step_times = []
    pbar = tqdm(total=test_episode)
    while total < 1000:
        if perlin:
          addPerlinNoiseToObs(obs, perlin)
          addPerlinNoiseToInHand(in_hands, perlin)
        q_value_maps, actions_star_idx, actions_star = agent.getEGreedyActions(states, in_hands, obs, 0, 0)
        # plt.imshow(obs[0, 0])
        # plt.show()
        # plotQMaps(q_value_maps)
        # plotSoftmax(q_value_maps)
        # plotQMaps(q_value_maps, save='/media/dian/hdd/analysis/qmap/house1_dqfdall', j=j)
        # plotSoftmax(q_value_maps, save='/media/dian/hdd/analysis/qmap/house1_dqfd_400k', j=j)
        actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)

        # plan_actions = envs.getPlan()
        # planner_actions_star_idx, planner_actions_star = agent.getActionFromPlan(plan_actions)
        # ranks.extend(rankOfAction(q_value_maps, planner_actions_star_idx))
        # print('avg rank of ae: {}'.format(np.mean(ranks)))

        states_, in_hands_, obs_, rewards, dones = envs.step(actions_star, auto_reset=True)
        obs_ = obs_.permute(0, 3, 1, 2)
        in_hands_ = in_hands_.permute(0, 3, 1, 2)

        states = copy.copy(states_)
        obs = copy.copy(obs_)
        in_hands = in_hands_

        s += rewards.sum().int().item()

        if dones.sum():
            total += dones.sum().int().item()

        pbar.set_description(
            '{}/{}, SR: {:.3f}'
                .format(s, total, float(s) / total if total != 0 else 0)
        )
        pbar.update(dones.sum().int().item())

    # np.save('ranks_dqfd_all.npy', ranks)
    # plotRanks(ranks, 1200)

if __name__ == '__main__':
    test()
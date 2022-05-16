import time
import numpy as np
import torch
import copy

from tqdm import tqdm

from bulletarm_baselines.fc_dqn.scripts.create_agent import createAgent
from bulletarm_baselines.fc_dqn.utils.env_wrapper import EnvWrapper
from bulletarm_baselines.fc_dqn.utils.parameters import *
from bulletarm_baselines.fc_dqn.scripts.create_agent import createAgent

def testPlanner():
    # test_episode = 500
    test_episode = 1000
    envs = EnvWrapper(num_processes, env, env_config, planner_config)
    agent = createAgent()
    agent.eval()
    states, in_hands, obs = envs.reset()
    total = 0
    s = 0
    step_times = []
    pbar = tqdm(total=test_episode)
    while total < test_episode:
        plan_actions = envs.getNextAction()
        actions_star_idx, actions_star = agent.getActionFromPlan(plan_actions)
        actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)
        t0 = time.time()
        states_, in_hands_, obs_, rewards, dones = envs.step(actions_star, auto_reset=True)
        t = time.time()-t0
        step_times.append(t)
        s += rewards.sum().int().item()

        if dones.sum():
            total += dones.sum().int().item()
            # print('{}/{}'.format(s, total))

        pbar.set_description(
            '{}/{}, SR: {:.3f}, step time: {:.2f}; avg step time: {:.2f}'
            .format(s, total, float(s)/total if total !=0 else 0, t, np.mean(step_times))
        )
        pbar.update(dones.sum().int().item())
        obs_ = obs_.permute(0, 3, 1, 2)
        states = copy.copy(states_)
        obs = copy.copy(obs_)

if __name__ == '__main__':
    testPlanner()
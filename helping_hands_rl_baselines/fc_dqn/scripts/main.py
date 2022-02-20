import os
import sys
import time
import copy
import math
import collections
from tqdm import tqdm
import datetime

import torch

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
sys.path.append('./')
sys.path.append('..')
from helping_hands_rl_baselines.fc_dqn.scripts.create_agent import createAgent
from helping_hands_rl_baselines.fc_dqn.storage.buffer import QLearningBufferExpert, QLearningBuffer
# from helping_hands_rl_baselines.fc_dqn.utils.logger import Logger
from helping_hands_rl_baselines.logger.baseline_logger import BaselineLogger
from helping_hands_rl_baselines.fc_dqn.utils.schedules import LinearSchedule
from helping_hands_rl_baselines.fc_dqn.utils.env_wrapper import EnvWrapper

from helping_hands_rl_baselines.fc_dqn.utils.parameters import *
from helping_hands_rl_baselines.fc_dqn.utils.torch_utils import augmentBuffer, augmentBufferD4
from helping_hands_rl_baselines.fc_dqn.scripts.fill_buffer_deconstruct import fillDeconstruct


ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done step_left expert')

def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def getCurrentObs(in_hand, obs):
    obss = []
    for i, o in enumerate(obs):
        obss.append((o.squeeze(), in_hand[i].squeeze()))
    return obss

def train_step(agent, replay_buffer, logger):
    batch = replay_buffer.sample(batch_size)
    loss, td_error = agent.update(batch)
    logger.logTrainingStep(loss)
    logger.num_training_steps += 1
    if logger.num_training_steps % target_update_freq == 0:
        agent.updateTarget()

def saveModelAndInfo(logger, agent):
    logger.writeLog()
    logger.exportData()

def train():
    start_time = time.time()
    if seed is not None:
        set_seed(seed)
    # setup env
    envs = EnvWrapper(num_processes,  env, env_config, planner_config)

    # setup agent
    agent = createAgent()

    if load_model_pre:
        agent.loadModel(load_model_pre)
    agent.train()

    # logging
    log_dir = os.path.join(log_pre, '{}_{}_{}'.format(alg, model, env))
    if note:
        log_dir += '_'
        log_dir += note
    if not log_sub:
        timestamp = time.time()
        timestamp = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d.%H:%M:%S')
        log_dir = os.path.join(log_dir, timestamp)
    else:
        log_dir = os.path.join(log_dir, log_sub)

    # logger = Logger(log_dir, env, 'train', num_processes, max_episode, log_sub)

    hyper_parameters['model_shape'] = agent.getModelStr()
    logger = BaselineLogger(log_dir, hyperparameters=hyper_parameters)
    logger.saveParameters(hyper_parameters)

    if buffer_type == 'expert':
        replay_buffer = QLearningBufferExpert(buffer_size)
    else:
        replay_buffer = QLearningBuffer(buffer_size)
    exploration = LinearSchedule(schedule_timesteps=explore, initial_p=init_eps, final_p=final_eps)

    states, in_hands, obs = envs.reset()

    if load_sub:
        logger.loadCheckPoint(os.path.join(log_dir, load_sub, 'checkpoint'), agent, replay_buffer)

    if planner_episode > 0 and not load_sub:
        if fill_buffer_deconstruct:
            fillDeconstruct(agent, replay_buffer)
        else:
            planner_envs = envs
            planner_num_process = num_processes
            j = 0
            states, obs = planner_envs.reset()
            s = 0
            if not no_bar:
                planner_bar = tqdm(total=planner_episode)
            while j < planner_episode:
                plan_actions = planner_envs.getNextAction()
                planner_actions_star_idx, planner_actions_star = agent.getActionFromPlan(plan_actions)
                states_, obs_, rewards, dones = planner_envs.step(planner_actions_star, auto_reset=True)
                steps_lefts = planner_envs.getStepLeft()
                for i in range(planner_num_process):
                    transition = ExpertTransition(states[i].numpy(), obs[i].numpy(), planner_actions_star_idx[i].numpy(),
                                                  rewards[i].numpy(), states_[i].numpy(), obs_[i].numpy(), dones[i].numpy(),
                                                  steps_lefts[i].numpy(), np.array(1))
                    replay_buffer.add(transition)
                states = copy.copy(states_)
                obs = copy.copy(obs_)

                j += dones.sum().item()
                s += rewards.sum().item()

                if not no_bar:
                    planner_bar.set_description('{:.3f}/{}, AVG: {:.3f}'.format(s, j, float(s)/j if j != 0 else 0))
                    planner_bar.update(dones.sum().item())
        if expert_aug_n > 0:
            augmentBuffer(replay_buffer, expert_aug_n, agent.rzs)
        elif expert_aug_d4:
            augmentBufferD4(replay_buffer, agent.rzs)

    # pre train
    if pre_train_step > 0:
        pbar = tqdm(total=pre_train_step)
        while logger.num_training_steps < pre_train_step:
            t0 = time.time()
            train_step(agent, replay_buffer, logger)
            if not no_bar:
                pbar.set_description('loss: {:.3f}, time: {:.2f}'.format(float(logger.getCurrentLoss()), time.time()-t0))
                pbar.update(len(logger.num_training_steps)-pbar.n)

            if (time.time() - start_time) / 3600 > time_limit:
                logger.saveCheckPoint(args, agent, replay_buffer)
                exit(0)
        pbar.close()
        logger.saveModel('pretrain', agent)
        # agent.sl = sl

    if not no_bar:
        pbar = tqdm(total=max_episode)
        pbar.set_description('Episodes:0; Reward:0.0; Explore:0.0; Loss:0.0; Time:0.0')
    timer_start = time.time()

    while logger.num_eps < max_episode:
        if fixed_eps:
            eps = final_eps
        else:
            eps = exploration.value(logger.num_eps)
        is_expert = 0
        q_value_maps, actions_star_idx, actions_star = agent.getEGreedyActions(states, in_hands, obs, eps)

        buffer_obs = getCurrentObs(in_hands, obs)
        actions_star = torch.cat((actions_star, states.unsqueeze(1)), dim=1)
        envs.stepAsync(actions_star, auto_reset=False)

        if len(replay_buffer) >= training_offset:
            for training_iter in range(training_iters):
                train_step(agent, replay_buffer, logger)

        states_, in_hands_, obs_, rewards, dones = envs.stepWait()
        steps_lefts = envs.getStepLeft()

        done_idxes = torch.nonzero(dones).squeeze(1)
        if done_idxes.shape[0] != 0:
            reset_states_, reset_in_hands_, reset_obs_ = envs.reset_envs(done_idxes)
            for j, idx in enumerate(done_idxes):
                states_[idx] = reset_states_[j]
                in_hands_[idx] = reset_in_hands_[j]
                obs_[idx] = reset_obs_[j]

        buffer_obs_ = getCurrentObs(in_hands_, obs_)

        for i in range(num_processes):
            replay_buffer.add(
                ExpertTransition(states[i], buffer_obs[i], actions_star_idx[i], rewards[i], states_[i],
                                 buffer_obs_[i], dones[i], steps_lefts[i], torch.tensor(is_expert))
            )
        logger.logStep(rewards.numpy(), dones.numpy())

        states = copy.copy(states_)
        obs = copy.copy(obs_)
        in_hands = copy.copy(in_hands_)

        if (time.time() - start_time)/3600 > time_limit:
            break

        if not no_bar:
            timer_final = time.time()
            description = 'Steps:{}; Reward:{:.03f}; Explore:{:.02f}; Loss:{:.03f}; Time:{:.03f}'.format(
                logger.num_steps, logger.getAvg(logger.training_eps_rewards, 1000), eps, float(logger.getCurrentLoss()),
                timer_final - timer_start)
            pbar.set_description(description)
            timer_start = timer_final
            pbar.update(logger.num_eps-pbar.n)
        logger.num_steps += num_processes

        if logger.num_steps % (num_processes * save_freq) == 0:
            saveModelAndInfo(logger, agent)

    saveModelAndInfo(logger, agent)
    logger.saveCheckPoint(args, agent, replay_buffer)
    envs.close()

if __name__ == '__main__':
    train()
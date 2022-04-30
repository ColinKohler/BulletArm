import os
import sys
import time
import copy
import collections
from tqdm import tqdm
import datetime

# sys.path.append('./')
# sys.path.append('..')
sys.path.append('../../..')
from helping_hands_rl_baselines.equi_rl.utils.parameters import *
from helping_hands_rl_baselines.equi_rl.storage.buffer import QLearningBufferExpert, QLearningBuffer
from helping_hands_rl_baselines.equi_rl.storage.per_buffer import PrioritizedQLearningBuffer, EXPERT, NORMAL
from helping_hands_rl_baselines.equi_rl.storage.aug_buffer import QLearningBufferAug
from helping_hands_rl_baselines.equi_rl.storage.per_aug_buffer import PrioritizedQLearningBufferAug
from helping_hands_rl_baselines.equi_rl.utils.schedules import LinearSchedule
from helping_hands_rl_baselines.equi_rl.utils.env_wrapper import EnvWrapper

from helping_hands_rl_baselines.equi_rl.utils.create_agent import createAgent
import threading

from helping_hands_rl_baselines.equi_rl.utils.torch_utils import ExpertTransition, normalizeTransition, augmentBuffer

from helping_hands_rl_baselines.logger.baseline_logger import BaselineLogger

def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_step(agent, replay_buffer, logger, p_beta_schedule):
    if buffer_type[:3] == 'per':
        beta = p_beta_schedule.value(logger.num_training_steps)
        batch, weights, batch_idxes = replay_buffer.sample(batch_size, beta)
        loss, td_error = agent.update(batch)
        new_priorities = np.abs(td_error.cpu()) + np.stack([t.expert for t in batch]) * per_expert_eps + per_eps
        replay_buffer.update_priorities(batch_idxes, new_priorities)
    else:
        batch = replay_buffer.sample(batch_size)
        loss, td_error = agent.update(batch)

    logger.logTrainingStep(loss)
    if logger.num_training_steps % target_update_freq == 0:
        agent.updateTarget()

def preTrainCURLStep(agent, replay_buffer, logger):
    if buffer_type[:3] == 'per':
        batch, weights, batch_idxes = replay_buffer.sample(batch_size, per_beta)
    else:
        batch = replay_buffer.sample(batch_size)
    agent.updateCURLOnly(batch)

def saveModelAndInfo(logger, agent):
    logger.writeLog()
    logger.exportData()
    agent.saveModel(os.path.join(logger.models_dir, 'snapshot'))


def evaluate(envs, agent, logger):
    states, obs = envs.reset()
    evaled = 0
    temp_reward = [[] for _ in range(num_eval_processes)]
    if not no_bar:
        eval_bar = tqdm(total=num_eval_episodes)
    while evaled < num_eval_episodes:
        actions_star_idx, actions_star = agent.getGreedyActions(states, obs)
        states_, obs_, rewards, dones = envs.step(actions_star, auto_reset=True)
        rewards = rewards.numpy()
        dones = dones.numpy()
        states = copy.copy(states_)
        obs = copy.copy(obs_)
        for i, r in enumerate(rewards.reshape(-1)):
            temp_reward[i].append(r)
        evaled += int(np.sum(dones))
        for i, d in enumerate(dones.astype(bool)):
            if d:
                R = 0
                for r in reversed(temp_reward[i]):
                    R = r + gamma * R
                logger.logEvalEpisode(temp_reward[i], discounted_return=R)
                temp_reward[i] = []
        if not no_bar:
            eval_bar.update(evaled - eval_bar.n)
    # logger.eval_rewards.append(np.mean(eval_rewards[:num_eval_episodes]))
    logger.logEvalInterval()
    logger.writeLog()
    if not no_bar:
        eval_bar.close()

def countParameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def train():
    eval_thread = None
    start_time = time.time()
    if seed is not None:
        set_seed(seed)
    # setup env
    print('creating envs')
    envs = EnvWrapper(num_processes, simulator, env, env_config, planner_config)
    eval_envs = EnvWrapper(num_eval_processes, simulator, env, env_config, planner_config)

    # setup agent
    agent = createAgent()
    eval_agent = createAgent(test=True)
    # .train() is required for equivariant network
    agent.train()
    eval_agent.train()
    if load_model_pre:
        agent.loadModel(load_model_pre)

    # logging
    base_dir = os.path.join(log_pre, '{}_{}_{}'.format(alg, model, env))
    if note:
        base_dir += '_'
        base_dir += note
    if not log_sub:
        timestamp = time.time()
        timestamp = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d.%H:%M:%S')
        log_dir = os.path.join(base_dir, timestamp)
    else:
        log_dir = os.path.join(base_dir, log_sub)

    hyper_parameters['model_shape'] = agent.getModelStr()
    logger = BaselineLogger(log_dir, checkpoint_interval=save_freq, num_eval_eps=num_eval_episodes, hyperparameters=hyper_parameters, eval_freq=eval_freq)
    logger.saveParameters(hyper_parameters)

    if buffer_type == 'per':
        replay_buffer = PrioritizedQLearningBuffer(buffer_size, per_alpha, NORMAL)
    elif buffer_type == 'per_expert':
        replay_buffer = PrioritizedQLearningBuffer(buffer_size, per_alpha, EXPERT)
    elif buffer_type == 'expert':
        replay_buffer = QLearningBufferExpert(buffer_size)
    elif buffer_type == 'normal':
        replay_buffer = QLearningBuffer(buffer_size)
    elif buffer_type == 'aug':
        replay_buffer = QLearningBufferAug(buffer_size, aug_n=buffer_aug_n)
    elif buffer_type == 'per_expert_aug':
        replay_buffer = PrioritizedQLearningBufferAug(buffer_size, per_alpha, EXPERT, aug_n=buffer_aug_n)
    else:
        raise NotImplementedError
    exploration = LinearSchedule(schedule_timesteps=explore, initial_p=init_eps, final_p=final_eps)
    p_beta_schedule = LinearSchedule(schedule_timesteps=max_train_step, initial_p=per_beta, final_p=1.0)

    if load_sub:
        logger.loadCheckPoint(os.path.join(base_dir, load_sub, 'checkpoint'), agent.loadFromState, replay_buffer.loadFromState)

    if planner_episode > 0 and not load_sub:
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
                if obs_type == 'pixel':
                    transition = normalizeTransition(transition)
                replay_buffer.add(transition)
            states = copy.copy(states_)
            obs = copy.copy(obs_)

            j += dones.sum().item()
            s += rewards.sum().item()

            if not no_bar:
                planner_bar.set_description('{:.3f}/{}, AVG: {:.3f}'.format(s, j, float(s)/j if j != 0 else 0))
                planner_bar.update(dones.sum().item())
        if expert_aug_n > 0:
            augmentBuffer(replay_buffer, buffer_aug_type, expert_aug_n)

        if alg in ['curl_sac', 'curl_sacfd', 'curl_sacfd_mean']:
            if not no_bar:
                pre_train_bar = tqdm(total=1600)
            while j < 1600:
                preTrainCURLStep(agent, replay_buffer, logger)
                j += 1
                if not no_bar:
                    pre_train_bar.update(1)

    # pre train
    if pre_train_step > 0:
        pbar = tqdm(total=pre_train_step)
        while logger.num_training_steps < pre_train_step:
            t0 = time.time()
            train_step(agent, replay_buffer, logger, p_beta_schedule)
            if logger.num_training_steps % 1000 == 0:
                logger.saveLossCurve(100)
            if not no_bar:
                pbar.set_description('loss: {:.3f}, time: {:.2f}'.format(float(logger.getCurrentLoss()), time.time()-t0))
                pbar.update(len(logger.num_training_steps)-pbar.n)

            if (time.time() - start_time) / 3600 > time_limit:
                logger.saveCheckPoint(agent.getSaveState(), replay_buffer.getSaveState())
                exit(0)
        pbar.close()

    if not no_bar:
        pbar = tqdm(total=max_train_step)
        pbar.set_description('Episodes:0; Reward:0.0; Explore:0.0; Loss:0.0; Time:0.0')
    timer_start = time.time()

    states, obs = envs.reset()
    while logger.num_training_steps < max_train_step:
        if fixed_eps:
            eps = final_eps
        else:
            eps = exploration.value(logger.num_training_steps)

        is_expert = 0
        actions_star_idx, actions_star = agent.getEGreedyActions(states, obs, eps)

        envs.stepAsync(actions_star, auto_reset=False)

        if len(replay_buffer) >= training_offset:
            for training_iter in range(training_iters):
                train_step(agent, replay_buffer, logger, p_beta_schedule)

        states_, obs_, rewards, dones = envs.stepWait()
        steps_lefts = envs.getStepLeft()

        done_idxes = torch.nonzero(dones).squeeze(1)
        if done_idxes.shape[0] != 0:
            reset_states_, reset_obs_ = envs.reset_envs(done_idxes)
            for j, idx in enumerate(done_idxes):
                states_[idx] = reset_states_[j]
                obs_[idx] = reset_obs_[j]

        if not alg[:2] == 'bc':
            for i in range(num_processes):
                transition = ExpertTransition(states[i].numpy(), obs[i].numpy(), actions_star_idx[i].numpy(),
                                              rewards[i].numpy(), states_[i].numpy(), obs_[i].numpy(), dones[i].numpy(),
                                              steps_lefts[i].numpy(), np.array(is_expert))
                if obs_type == 'pixel':
                    transition = normalizeTransition(transition)
                replay_buffer.add(transition)
        logger.logStep(rewards.numpy(), dones.numpy())

        states = copy.copy(states_)
        obs = copy.copy(obs_)

        if (time.time() - start_time)/3600 > time_limit:
            break

        if not no_bar:
            timer_final = time.time()
            description = 'Action Step:{}; Reward:{:.03f}; Eval Reward:{:.03f}; Explore:{:.02f}; Loss:{:.03f}; Time:{:.03f}'.format(
                logger.num_steps, logger.getAvg(logger.training_eps_rewards, 100), np.mean(logger.eval_eps_rewards[-2]) if len(logger.eval_eps_rewards) > 1 and len(logger.eval_eps_rewards[-2]) > 0 else 0, eps, float(logger.getCurrentLoss()),
                timer_final - timer_start)
            pbar.set_description(description)
            timer_start = timer_final
            pbar.update(logger.num_training_steps-pbar.n)

        if logger.num_training_steps > 0 and eval_freq > 0 and logger.num_training_steps % eval_freq == 0:
            if eval_thread is not None:
                eval_thread.join()
            eval_agent.copyNetworksFrom(agent)
            eval_thread = threading.Thread(target=evaluate, args=(eval_envs, eval_agent, logger))
            eval_thread.start()
            # evaluate(eval_envs, agent, logger)

        if logger.num_steps % (num_processes * save_freq) == 0:
            saveModelAndInfo(logger, agent)

    if eval_thread is not None:
        eval_thread.join()
    saveModelAndInfo(logger, agent)
    logger.saveCheckPoint(agent.getSaveState(), replay_buffer.getSaveState())
    envs.close()
    eval_envs.close()
    print('training finished')
    if not no_bar:
        pbar.close()

if __name__ == '__main__':
    train()
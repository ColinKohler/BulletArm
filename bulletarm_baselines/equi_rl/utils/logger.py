import os
import time
import datetime
import math
import numpy as np
:mport matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import namedtuple
from more_itertools import windowed
import dill as pickle
import json
from tqdm import tqdm

import torch
import torch.nn.functional as F

from bulletarm_baselines.equi_rl.utils.parameters import *

# Transition object
from bulletarm_baselines.equi_rl.utils.torch_utils import ExpertTransition

plt.style.use('ggplot')

class Logger(object):
    '''
    Logger for train/test runs.

    Args:
      - log_dir: Directory to write log
      - num_envs: Number of environments running concurrently
    '''

    def __init__(self, log_dir, env, mode, num_envs, max_train_step, gamma, log_dir_sub=None):
        # Logging variables
        self.env = env
        self.mode = mode
        self.max_train_step = max_train_step
        self.num_envs = num_envs
        self.gamma = gamma

        # Create directory in the logging directory
        timestamp = time.time()
        timestamp = datetime.datetime.fromtimestamp(timestamp)
        if not log_dir_sub:
            self.base_dir = os.path.join(log_dir, '{}_{}_{}'.format(self.mode, self.env, timestamp.strftime('%Y-%m-%d.%H:%M:%S')))
        else:
            self.base_dir = os.path.join(log_dir, log_dir_sub)
        print('Creating logging session at: {}'.format(self.base_dir))

        # Create subdirs to save important run info
        self.info_dir = os.path.join(self.base_dir, 'info')
        self.models_dir = os.path.join(self.base_dir, 'models')
        self.checkpoint_dir = os.path.join(self.base_dir, 'checkpoint')

        os.makedirs(self.info_dir)
        os.makedirs(self.models_dir)
        os.makedirs(self.checkpoint_dir)

        # Variables to hold episode information
        # self.episode_rewards = np.zeros(self.num_envs)
        self.episode_rewards = [[] for _ in range(self.num_envs)]
        self.num_steps = 0
        self.num_training_steps = 0
        self.num_episodes = 0
        self.rewards = list()
        self.losses = list()
        self.steps_left = list()
        self.td_errors = list()
        self.expert_samples = list()

        self.eval_rewards = list()

        # Buffer of transitions
        self.transitions = list()

    def stepBookkeeping(self, rewards, step_lefts, done_masks):
        for i, r in enumerate(rewards.reshape(-1)):
            self.episode_rewards[i].append(r)
        # self.episode_rewards += rewards.squeeze()
        self.num_episodes += int(np.sum(done_masks))
        for i, d in enumerate(done_masks.astype(bool)):
            if d:
                R = 0
                for r in reversed(self.episode_rewards[i]):
                    R = r + self.gamma * R
                self.rewards.append(R)
                self.episode_rewards[i] = []
        # self.rewards.extend(self.episode_rewards[done_masks.astype(bool)])
        self.steps_left.extend(step_lefts[done_masks.astype(bool)])
        # self.episode_rewards[done_masks.astype(bool)] = 0.

    def trainingBookkeeping(self, loss, td_error):
        self.losses.append(loss)
        self.td_errors.append(td_error)

    def tdErrorBookkeeping(self, td_error):
        self.td_errors.append(td_error)

    def close(self):
        ''' Close the logger and save the logging information '''
        # self.saveLearningCurve()
        # self.saveLossCurve()
        self.saveRewards()
        self.saveLosses()
        self.saveTdErrors()

    def getCurrentAvgReward(self, n=100, starting=0):
        ''' Get the average reward for the last n episodes '''
        if not self.rewards:
            return 0.0
        starting = max(starting, len(self.rewards)-n)
        return np.mean(self.rewards[starting:])
        # return np.mean(self.rewards[-n:]) if self.rewards else 0.0

    def getCurrentLoss(self):
        ''' Get the most recent loss. '''
        if not self.losses:
            return 0.0
        current_loss = self.losses[-1]
        if type(current_loss) is float:
            return current_loss
        else:
            return np.mean(current_loss)

    def saveLearningCurve(self, n=100):
        ''' Plot the rewards over timesteps and save to logging dir '''
        n = min(n, len(self.rewards))
        if n > 0:
            avg_reward = np.mean(list(windowed(self.rewards, n)), axis=1)
            xs = np.arange(n, (len(avg_reward))+n)
            plt.plot(xs, np.mean(list(windowed(self.rewards, n)), axis=1))
            plt.savefig(os.path.join(self.info_dir, 'learning_curve.pdf'))
            plt.close()

    def saveStepLeftCurve(self, n=100):
        n = min(n, len(self.steps_left))
        if n > 0:
            plt.plot(np.mean(list(windowed(self.steps_left, n)), axis=1))
            plt.savefig(os.path.join(self.info_dir, 'steps_left_curve.pdf'))
            plt.close()

    def saveLossCurve(self, n=100):
        losses = np.array(self.losses)
        if len(losses) < n:
            return
        if len(losses.shape) == 1:
            losses = np.expand_dims(losses, 0)
        else:
            losses = np.moveaxis(losses, 1, 0)
        for loss in losses:
            plt.plot(np.mean(list(windowed(loss, n)), axis=1))

        plt.savefig(os.path.join(self.info_dir, 'loss_curve.pdf'))
        plt.yscale('log')
        plt.savefig(os.path.join(self.info_dir, 'loss_curve_log.pdf'))

        plt.close()

    def saveTdErrorCurve(self, n=100):
        n = min(n, len(self.td_errors))
        if n > 0:
            plt.plot(np.mean(list(windowed(self.td_errors, n)), axis=1))
            plt.yscale('log')
            plt.savefig(os.path.join(self.info_dir, 'td_error_curve.pdf'))
            plt.close()

    def saveEvalCurve(self):
        if len(self.eval_rewards) > 0:
            xs = np.arange(eval_freq, (len(self.eval_rewards)+1) * eval_freq, eval_freq)
            plt.plot(xs, self.eval_rewards)
            plt.savefig(os.path.join(self.info_dir, 'eval_curve.pdf'))
            plt.close()

    def saveModel(self, iteration, name, agent):
        '''
        Save PyTorch model to log directory

        Args:
          - iteration: Interation of the current run
          - name: Name to save model as
          - agent: Agent containing model to save
        '''
        agent.saveModel(os.path.join(self.models_dir, 'snapshot_{}'.format(name)))

    def saveRewards(self):
        np.save(os.path.join(self.info_dir, 'rewards.npy'), self.rewards)

    def saveLosses(self):
        np.save(os.path.join(self.info_dir, 'losses.npy'), self.losses)

    def saveTdErrors(self):
        np.save(os.path.join(self.info_dir, 'td_errors.npy'), self.td_errors)

    def saveCandidateSchedule(self, schedule):
        np.save(os.path.join(self.info_dir, 'schedule.npy'), schedule)

    def saveEvalRewards(self):
        np.save(os.path.join(self.info_dir, 'eval_rewards.npy'), self.eval_rewards)

    def saveParameters(self, parameters):
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)
        with open(os.path.join(self.info_dir, "parameters.json"), 'w') as f:
            json.dump(parameters, f, cls=NumpyEncoder)

    def saveBuffer(self, buffer):
        print('saving buffer')
        torch.save(buffer.getSaveState(), os.path.join(self.checkpoint_dir, 'buffer.pt'))

    def loadBuffer(self, buffer, path, max_n=1000000):
        print('loading buffer: '+path)
        load = torch.load(path)
        if not no_bar:
            loop = tqdm(range(len(load['storage'])))
        else:
            loop = range(len(load['storage']))
        for i in loop:
            if i == max_n:
                break
            t = load['storage'][i]
            buffer.add(t)

    def saveCheckPoint(self, args, envs, agent, buffer):
        envs_save_path = os.path.join(self.checkpoint_dir, 'envs')
        envs.saveToFile(envs_save_path)

        checkpoint = {
            'args': args.__dict__,
            'agent': agent.getSaveState(),
            'buffer_state': buffer.getSaveState(),
            'logger':{
                'env': self.env,
                'num_envs': self.num_envs,
                'max_train_step': self.max_train_step,
                'episode_rewards': self.episode_rewards,
                'num_steps': self.num_steps,
                'num_training_steps': self.num_training_steps,
                'num_episodes': self.num_episodes,
                'rewards': self.rewards,
                'losses': self.losses,
                'steps_left': self.steps_left,
                'td_errors': self.td_errors,
                'expert_samples': self.expert_samples,
                'eval_rewards': self.eval_rewards,
            },
            'torch_rng_state': torch.get_rng_state(),
            'torch_cuda_rng_state': torch.cuda.get_rng_state(),
            'np_rng_state': np.random.get_state()
        }
        if hasattr(agent, 'his'):
            checkpoint.update({'agent_his': agent.his})
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'checkpoint.pt'))

    def loadCheckPoint(self, checkpoint_dir, envs, agent, buffer):
        print('loading checkpoint')
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'))
        print('agent loading')
        agent.loadFromState(checkpoint['agent'])
        print('buffer loading')
        buffer.loadFromState(checkpoint['buffer_state'])
        print('logger loading')
        self.env = checkpoint['logger']['env']
        self.num_envs = checkpoint['logger']['num_envs']
        # self.max_episode = checkpoint['logger']['max_episode']
        self.episode_rewards = checkpoint['logger']['episode_rewards']
        self.num_steps = checkpoint['logger']['num_steps']
        self.num_training_steps = checkpoint['logger']['num_training_steps']
        self.num_episodes = checkpoint['logger']['num_episodes']
        self.rewards = checkpoint['logger']['rewards']
        self.losses = checkpoint['logger']['losses']
        self.steps_left = checkpoint['logger']['steps_left']
        self.td_errors =checkpoint['logger']['td_errors']
        self.expert_samples = checkpoint['logger']['expert_samples']
        self.eval_rewards = checkpoint['logger']['eval_rewards']
        torch.set_rng_state(checkpoint['torch_rng_state'])
        torch.cuda.set_rng_state(checkpoint['torch_cuda_rng_state'])
        np.random.set_state(checkpoint['np_rng_state'])

        if hasattr(agent, 'his'):
            agent.his = checkpoint['agent_his']

        # envs_save_path = os.path.join(checkpoint_dir, 'envs')
        # success = envs.loadFromFile(envs_save_path)
        # if not success:
        #   raise EnvironmentError

        print('loaded checkpoint')

    def expertSampleBookkeeping(self, expert_ratio):
        self.expert_samples.append(expert_ratio)


    def saveExpertSampleCurve(self, n=100):
        n = min(n, len(self.expert_samples))
        if n > 0:
            plt.plot(np.mean(list(windowed(self.expert_samples, n)), axis=1))
            plt.savefig(os.path.join(self.info_dir, 'expert_sample_curve.pdf'))
            plt.close()

    def saveResult(self):
        result_dir = os.path.join(self.base_dir, 'result')
        os.makedirs(result_dir)
        np.save(os.path.join(result_dir, 'rewards.npy'), self.rewards)
        np.save(os.path.join(result_dir, 'losses.npy'), self.losses)
        np.save(os.path.join(result_dir, 'td_errors.npy'), self.td_errors)
        np.save(os.path.join(result_dir, 'eval_rewards.npy'), self.eval_rewards)
